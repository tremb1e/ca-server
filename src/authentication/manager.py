from __future__ import annotations

import asyncio
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..ca_config import get_ca_config
from ..config import settings
from ..storage.inference_storage import InferenceStorage
from ..utils.reject_trackers import ConsecutiveRejectTracker, VoteRejectTracker

if TYPE_CHECKING:
    import torch

    from .runner import AuthRunConfig
    from .vqgan_inference import VQGANPolicy


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
    policy: "VQGANPolicy"
    created_at: float = field(default_factory=time.time)
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
        self._cache: Dict[str, "torch.nn.Module"] = {}
        self._order: List[str] = []

    def get(self, policy: "VQGANPolicy") -> "torch.nn.Module":
        from ..utils.accelerator import resolve_torch_device
        from .vqgan_inference import load_vqgan

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

    def snapshot(self) -> Dict[str, Any]:
        return {
            "max_models": int(self._max_models),
            "loaded_count": int(len(self._cache)),
            "device": str(self._device),
            "keys": list(self._order),
        }


class AuthSessionManager:
    def __init__(
        self,
        *,
        max_cached_models: int = 4,
        session_ttl_sec: int = 600,
        max_concurrent_inference: Optional[int] = None,
        models_root: Optional[Path] = None,
    ) -> None:
        self._sessions: Dict[str, AuthSessionState] = {}
        self._processing_cfg = None
        self._inference_storage = InferenceStorage(settings.inference_storage_path)
        self._model_cache = VQGANModelCache(max_models=max_cached_models)
        self._models_root = Path(models_root) if models_root is not None else Path(settings.data_storage_path).parent / "models"
        self._session_ttl_sec = int(session_ttl_sec)
        if max_concurrent_inference is None:
            max_concurrent_inference = min(8, os.cpu_count() or 1)
        max_concurrent_inference = int(max_concurrent_inference)
        if max_concurrent_inference <= 0:
            max_concurrent_inference = 1
        self._inference_semaphore = asyncio.Semaphore(max_concurrent_inference)

    @staticmethod
    def _session_key(user_id: str, session_id: str) -> str:
        return f"{user_id}\x1f{session_id}"

    def _policy_from_config(self, cfg: "AuthRunConfig") -> "VQGANPolicy":
        from .vqgan_inference import VQGANPolicy

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
        from .runner import load_best_policy

        try:
            cfg = load_best_policy(user_id, models_root=self._models_root)
        except Exception:
            return False
        if not cfg.vqgan_checkpoint.exists():
            return False
        if cfg.vqgan_config and not cfg.vqgan_config.exists():
            return False
        return True

    def start_session(self, user_id: str, session_id: str) -> Tuple[bool, str, Optional["VQGANPolicy"]]:
        self._prune_sessions()
        from .runner import load_best_policy

        try:
            cfg = load_best_policy(user_id, models_root=self._models_root)
        except Exception as exc:
            return False, f"model_not_ready: {exc}", None

        policy = self._policy_from_config(cfg)
        state = AuthSessionState(user_id=user_id, session_id=session_id, policy=policy)
        state.consecutive_rejects = ConsecutiveRejectTracker()
        state.vote_rejects = VoteRejectTracker()
        self._sessions[self._session_key(user_id, session_id)] = state
        return True, "ok", policy

    @staticmethod
    def _iso_from_epoch(value: float) -> str:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()

    @staticmethod
    def _policy_snapshot(policy: "VQGANPolicy") -> Dict[str, Any]:
        return {
            "user": str(policy.user),
            "window_size": float(policy.window_size),
            "overlap": float(policy.overlap),
            "target_width": int(policy.target_width),
            "threshold": float(policy.threshold),
            "k_rejects": int(policy.k_rejects),
            "vote_window_size": int(policy.vote_window_size),
            "vote_min_rejects": int(policy.vote_min_rejects),
            "model_version": str(policy.model_version),
            "vqgan_checkpoint": str(policy.vqgan_checkpoint),
            "vqgan_config": str(policy.vqgan_config),
        }

    def snapshot_sessions(self) -> List[Dict[str, Any]]:
        self._prune_sessions()
        sessions: List[Dict[str, Any]] = []
        for state in self._sessions.values():
            consecutive = state.consecutive_rejects
            vote = state.vote_rejects
            sessions.append(
                {
                    "user_id": str(state.user_id),
                    "session_id": str(state.session_id),
                    "created_at": self._iso_from_epoch(state.created_at),
                    "last_activity": self._iso_from_epoch(state.last_activity),
                    "idle_seconds": max(0.0, float(time.time() - state.last_activity)),
                    "window_index": int(state.window_index),
                    "tail_records": {k: len(v) for k, v in state.tail_records.items()},
                    "policy": self._policy_snapshot(state.policy),
                    "consecutive_rejects": {
                        "windows": int(getattr(consecutive, "windows", 0) or 0),
                        "rejects": int(getattr(consecutive, "rejects", 0) or 0),
                        "consecutive_rejects": int(getattr(consecutive, "consecutive_rejects", 0) or 0),
                        "interrupts": int(getattr(consecutive, "interrupts", 0) or 0),
                        "first_interrupt_window": getattr(consecutive, "first_interrupt_window", None),
                    },
                    "vote_rejects": {
                        "windows": int(getattr(vote, "windows", 0) or 0),
                        "rejects": int(getattr(vote, "rejects", 0) or 0),
                        "interrupts": int(getattr(vote, "interrupts", 0) or 0),
                        "first_interrupt_window": getattr(vote, "first_interrupt_window", None),
                        "recent_windows": int(getattr(vote, "recent_windows", 0) or 0),
                        "recent_rejects": int(getattr(vote, "recent_rejects", 0) or 0),
                    },
                }
            )
        return sessions

    def snapshot_model_cache(self) -> Dict[str, Any]:
        return self._model_cache.snapshot()

    async def handle_packet(
        self,
        *,
        user_id: str,
        session_id: str,
        parsed_batch: Dict[str, Any],
    ) -> Optional[AuthResultPayload]:
        self._prune_sessions()
        state = self._sessions.get(self._session_key(user_id, session_id))
        if state is None:
            return None

        state.last_activity = time.time()

        async with state.lock:
            from ..processing.pipeline import _extract_sensor_records, _resample_records, build_config
            from ..processing.scaler import apply_scaler, load_scaler
            from .vqgan_inference import score_windows, windowize_dataframe

            if self._processing_cfg is None:
                self._processing_cfg = build_config()

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
