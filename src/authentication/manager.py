from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch

from ..ca_config import get_ca_config
from ..config import settings
from ..processing.pipeline import _extract_sensor_records, _resample_records, build_config
from ..processing.scaler import apply_scaler, load_scaler
from ..storage.inference_storage import InferenceStorage
from ..utils.accelerator import resolve_torch_device
from ..utils.ca_train import ensure_ca_train_on_path
from .runner import AuthRunConfig, load_best_policy
from .vqgan_inference import VQGANPolicy, load_vqgan, score_windows, windowize_dataframe


@dataclass
class AuthResultPayload:
    user: str
    session_id: str
    window_id: int
    score: float
    threshold: float
    accept: bool
    interrupt: bool
    normalized_score: float
    k_rejects: int
    window_size: float
    model_version: str
    message: str


@dataclass
class AuthSessionState:
    user_id: str
    session_id: str
    policy: VQGANPolicy
    last_activity: float = field(default_factory=time.time)
    tail_records: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {"acc": [], "gyr": [], "mag": []})
    window_index: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    consecutive_rejects: Any = None
    vote_rejects: Any = None


class VQGANModelCache:
    def __init__(self, *, max_models: int = 4, device: Optional[str] = None) -> None:
        self._max_models = int(max_models)
        self._device = device or "auto"
        self._cache: Dict[str, torch.nn.Module] = {}
        self._order: List[str] = []

    def get(self, policy: VQGANPolicy) -> torch.nn.Module:
        key = f"{policy.user}::{policy.vqgan_checkpoint}"
        if key in self._cache:
            self._touch(key)
            return self._cache[key]

        if len(self._order) >= self._max_models:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)

        device = resolve_torch_device(self._device)
        model = load_vqgan(policy.vqgan_checkpoint, device=device, config_path=policy.vqgan_config)
        self._cache[key] = model
        self._order.append(key)
        return model

    def _touch(self, key: str) -> None:
        if key in self._order:
            self._order.remove(key)
        self._order.append(key)


class AuthSessionManager:
    def __init__(
        self,
        *,
        max_cached_models: int = 4,
        session_ttl_sec: int = 600,
        max_concurrent_inference: Optional[int] = None,
    ) -> None:
        self._sessions: Dict[str, AuthSessionState] = {}
        self._processing_cfg = build_config()
        self._inference_storage = InferenceStorage(settings.inference_storage_path)
        self._model_cache = VQGANModelCache(max_models=max_cached_models)
        self._session_ttl_sec = int(session_ttl_sec)
        if max_concurrent_inference is None:
            max_concurrent_inference = min(8, os.cpu_count() or 1)
        max_concurrent_inference = int(max_concurrent_inference)
        if max_concurrent_inference <= 0:
            max_concurrent_inference = 1
        self._inference_semaphore = asyncio.Semaphore(max_concurrent_inference)

    def _policy_from_config(self, cfg: AuthRunConfig) -> VQGANPolicy:
        ca_cfg = get_ca_config()
        vote_window_size = int(cfg.vote_window_size)
        vote_min_rejects = int(cfg.vote_min_rejects)
        if ca_cfg.auth.vote_window_size > 0 and ca_cfg.auth.vote_min_rejects > 0:
            vote_window_size = int(ca_cfg.auth.vote_window_size)
            vote_min_rejects = int(ca_cfg.auth.vote_min_rejects)
        return VQGANPolicy(
            user=cfg.user,
            window_size=cfg.window_size,
            overlap=cfg.overlap,
            target_width=cfg.target_width,
            threshold=cfg.threshold,
            k_rejects=cfg.k_rejects,
            vqgan_checkpoint=cfg.vqgan_checkpoint,
            vqgan_config=cfg.vqgan_config,
            vote_window_size=vote_window_size,
            vote_min_rejects=vote_min_rejects,
            model_version=cfg.model_version or cfg.vqgan_checkpoint.name,
        )

    @staticmethod
    def _format_vote_message(
        *,
        window_size: int,
        min_rejects: int,
        recent_windows: int,
        recent_rejects: int,
    ) -> str:
        if window_size <= 0 or min_rejects <= 0:
            return ""
        if recent_windows < window_size:
            return f"窗口不足 {recent_windows}/{window_size}，等待更多数据"
        return f"近{window_size}窗恶意 {recent_rejects}/{window_size}，阈值 {min_rejects}"

    def _prune_sessions(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if (now - v.last_activity) > self._session_ttl_sec]
        for key in expired:
            self._sessions.pop(key, None)

    def has_trained_model(self, user_id: str) -> bool:
        try:
            cfg = load_best_policy(user_id)
        except Exception:
            return False
        if not cfg.vqgan_checkpoint.exists():
            return False
        if cfg.vqgan_config and not cfg.vqgan_config.exists():
            return False
        return True

    def start_session(self, user_id: str, session_id: str) -> Tuple[bool, str, Optional[VQGANPolicy]]:
        self._prune_sessions()
        try:
            cfg = load_best_policy(user_id)
        except Exception as exc:
            return False, f"model_not_ready: {exc}", None

        policy = self._policy_from_config(cfg)
        ensure_ca_train_on_path()
        from hmog_consecutive_rejects import ConsecutiveRejectTracker, VoteRejectTracker  # type: ignore

        state = AuthSessionState(user_id=user_id, session_id=session_id, policy=policy)
        state.consecutive_rejects = ConsecutiveRejectTracker()
        state.vote_rejects = VoteRejectTracker()
        self._sessions[user_id] = state
        return True, "ok", policy

    async def handle_packet(
        self,
        *,
        user_id: str,
        session_id: str,
        parsed_batch: Dict[str, Any],
    ) -> Optional[AuthResultPayload]:
        self._prune_sessions()
        state = self._sessions.get(user_id)
        if state is None or state.session_id != session_id:
            return None

        state.last_activity = time.time()

        async with state.lock:
            await self._inference_storage.append_raw_packet(user_id, session_id, parsed_batch)

            packets = [{"sensor_batch": parsed_batch}]
            records = _extract_sensor_records(packets)
            combined_records = {
                k: (state.tail_records.get(k, []) + records.get(k, []))
                for k in ("acc", "gyr", "mag")
            }

            df = _resample_records(combined_records, session_label=session_id, user_id=user_id, cfg=self._processing_cfg)
            if df is None or df.empty:
                self._trim_tail(state, combined_records)
                return None

            scaler_path = Path(settings.processed_data_path) / "z-score" / user_id / "scaler.json"
            if not scaler_path.exists():
                return None
            scaler = load_scaler(scaler_path)
            normalized = apply_scaler(df, scaler)

            window_ids, windows = windowize_dataframe(
                normalized,
                window_size_sec=state.policy.window_size,
                overlap=state.policy.overlap,
                sampling_rate_hz=self._processing_cfg.sampling_rate_hz,
                target_width=state.policy.target_width,
            )

            if windows.size == 0:
                self._trim_tail(state, combined_records)
                return None

            model = self._model_cache.get(state.policy)
            device = next(model.parameters()).device
            async with self._inference_semaphore:
                scores = await asyncio.to_thread(
                    score_windows,
                    model,
                    windows,
                    device=device,
                    use_amp=True,
                )

            final_payload: Optional[AuthResultPayload] = None
            for offset, score in zip(window_ids, scores):
                window_id = state.window_index + int(offset)
                raw_score = float(score)
                accept = bool(raw_score >= float(state.policy.threshold))
                normalized_score = float(1.0 / (1.0 + math.exp(-raw_score)))

                interrupt = False
                decision_accept = accept
                decision_score = normalized_score
                decision_threshold = float(state.policy.threshold)
                decision_message = ""
                vote_recent_windows = 0
                vote_recent_rejects = 0
                if state.policy.vote_window_size > 0 and state.policy.vote_min_rejects > 0:
                    state.vote_rejects.update(
                        rejected=not accept,
                        window_size=int(state.policy.vote_window_size),
                        min_rejects=int(state.policy.vote_min_rejects),
                        reset_on_interrupt=False,
                    )
                    vote_recent_windows = int(state.vote_rejects.recent_windows)
                    vote_recent_rejects = int(state.vote_rejects.recent_rejects)
                    decision_ready = vote_recent_windows >= int(state.policy.vote_window_size)
                    decision_accept = not (
                        decision_ready and vote_recent_rejects >= int(state.policy.vote_min_rejects)
                    )
                    interrupt = not decision_accept
                    vote_window_size = float(state.policy.vote_window_size)
                    malicious_ratio = vote_recent_rejects / vote_window_size if vote_window_size > 0 else 0.0
                    decision_score = float(max(0.0, min(1.0, 1.0 - malicious_ratio)))
                    decision_threshold = float(
                        max(
                            0.0,
                            min(1.0, 1.0 - float(state.policy.vote_min_rejects) / vote_window_size),
                        )
                    )
                    decision_message = self._format_vote_message(
                        window_size=int(state.policy.vote_window_size),
                        min_rejects=int(state.policy.vote_min_rejects),
                        recent_windows=vote_recent_windows,
                        recent_rejects=vote_recent_rejects,
                    )
                elif state.policy.k_rejects > 0:
                    interrupt = state.consecutive_rejects.update(
                        rejected=not accept,
                        k=int(state.policy.k_rejects),
                        reset_on_interrupt=True,
                    )

                await self._inference_storage.append_result(
                    user_id,
                    session_id,
                    {
                        "window_id": window_id,
                        "score": raw_score,
                        "threshold": float(state.policy.threshold),
                        "accept": accept,
                        "interrupt": bool(interrupt),
                        "normalized_score": normalized_score,
                        "k_rejects": int(state.policy.k_rejects),
                        "vote_recent_windows": vote_recent_windows,
                        "vote_recent_rejects": vote_recent_rejects,
                        "window_size": float(state.policy.window_size),
                        "model_version": state.policy.model_version,
                    },
                )

                final_payload = AuthResultPayload(
                    user=user_id,
                    session_id=session_id,
                    window_id=window_id,
                    score=decision_score,
                    threshold=decision_threshold,
                    accept=decision_accept,
                    interrupt=bool(interrupt),
                    normalized_score=decision_score,
                    k_rejects=int(state.policy.k_rejects),
                    window_size=float(state.policy.window_size),
                    model_version=state.policy.model_version,
                    message=decision_message,
                )

            state.window_index += len(window_ids)
            self._trim_tail(state, combined_records)
            return final_payload

    def _trim_tail(self, state: AuthSessionState, records: Dict[str, List[Dict[str, Any]]]) -> None:
        tail_ms = int(round(float(state.policy.window_size) * float(state.policy.overlap) * 1000))
        if tail_ms <= 0:
            state.tail_records = {"acc": [], "gyr": [], "mag": []}
            return
        all_ts = [r["timestamp"] for lst in records.values() for r in lst if "timestamp" in r]
        if not all_ts:
            state.tail_records = {"acc": [], "gyr": [], "mag": []}
            return
        max_ts = max(all_ts)
        cutoff = max_ts - tail_ms
        trimmed = {}
        for sensor in ("acc", "gyr", "mag"):
            trimmed[sensor] = [r for r in records.get(sensor, []) if int(r.get("timestamp", 0)) >= cutoff]
        state.tail_records = trimmed
