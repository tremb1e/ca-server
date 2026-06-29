from __future__ import annotations

import asyncio
import hashlib
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ..ca_config import get_ca_config
from ..config import settings
from ..storage.inference_storage import InferenceStorage
from ..utils.reject_trackers import (
    ConsecutiveRejectTracker,
    EMAScoreTracker,
    ResultEmitGate,
    VoteRejectTracker,
    windows_for_delay,
)

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
    # 拆分尺度，避免混读：raw_* 为模型原始分数尺度，ema_* 为迟滞分数尺度，
    # display_* 为 0~1 可视化尺度。旧字段 score/threshold/normalized_score 保留为兼容字段。
    result_stage: str = "primary"
    raw_score: Optional[float] = None
    raw_threshold: Optional[float] = None
    ema_score: Optional[float] = None
    ema_threshold: Optional[float] = None
    display_score: Optional[float] = None
    display_threshold: Optional[float] = None


@dataclass
class AuthSessionState:
    user_id: str
    session_id: str
    policy: "VQGANPolicy"
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    tail_records: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {"acc": [], "gyr": [], "mag": []})
    window_index: int = 0
    # 认证结果发布节流闸门：按 auth.result_delay_sec 折算窗口数累积，满一个周期才发布一次结果。
    emit_gate: ResultEmitGate = field(default_factory=ResultEmitGate)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    consecutive_rejects: Any = None
    vote_rejects: Any = None
    ema_rejects: Any = None


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
        vote_window_size = int(cfg.vote_window_size or 0)
        vote_min_rejects = int(cfg.vote_min_rejects or 0)
        decision_strategy = str(cfg.decision_strategy or cfg.interrupt_rule or ca_cfg.auth.decision_strategy).strip().lower()
        if decision_strategy == "vote" and (vote_window_size <= 0 or vote_min_rejects <= 0):
            vote_window_size = int(ca_cfg.auth.vote_window_size)
            vote_min_rejects = int(ca_cfg.auth.vote_min_rejects)
        ema_alpha = float(cfg.ema_alpha or ca_cfg.auth.ema_alpha)
        return VQGANPolicy(
            user=cfg.user,
            window_size=cfg.window_size,
            overlap=cfg.overlap,
            target_width=cfg.target_width,
            threshold=cfg.threshold,
            interrupt_rule=cfg.interrupt_rule,
            decision_strategy=decision_strategy,
            k_rejects=cfg.k_rejects,
            vqgan_checkpoint=cfg.vqgan_checkpoint,
            vqgan_config=cfg.vqgan_config,
            vote_window_size=vote_window_size,
            vote_min_rejects=vote_min_rejects,
            ema_alpha=ema_alpha,
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

    @staticmethod
    def _artifact_fingerprint(path: Optional[Path]) -> Dict[str, Any]:
        """Return {path, exists, sha256(first 1MB or None), mtime(float or None)}."""
        fp: Dict[str, Any] = {
            "path": (str(path) if path is not None else None),
            "exists": False,
            "sha256": None,
            "mtime": None,
        }
        if path is None:
            return fp
        try:
            p = Path(path)
            if not p.exists() or not p.is_file():
                return fp
            fp["exists"] = True
            try:
                fp["mtime"] = float(p.stat().st_mtime)
            except Exception:
                fp["mtime"] = None
            try:
                with open(p, "rb") as fh:
                    chunk = fh.read(1024 * 1024)
                fp["sha256"] = hashlib.sha256(chunk).hexdigest()
            except Exception:
                fp["sha256"] = None
        except Exception:
            return fp
        return fp

    def _validate_model(self, user_id: str) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate a user's trained model + policy for authentication startup.

        Returns (ok, reason, details). The readiness gate accepts only the
        authoritative best policy (policy_status=ready, policy_search_completed)
        unless auth.allow_training_fallback_policy is enabled (degrade mode).
        """
        import json

        from .runner import load_best_policy

        allow_fb = bool(getattr(get_ca_config().auth, "allow_training_fallback_policy", False))
        try:
            cfg = load_best_policy(user_id, models_root=self._models_root, allow_training_fallback=allow_fb)
        except Exception as exc:
            return False, str(exc), {}

        policy_status = str(getattr(cfg, "policy_status", "") or "")
        policy_search_completed = bool(getattr(cfg, "policy_search_completed", False))
        policy_source = str(getattr(cfg, "policy_source", "") or "")
        policy_file = str(getattr(cfg, "policy_file", "") or "")
        score_metric = str(getattr(cfg, "score_metric", "") or "")
        score_scale = str(getattr(cfg, "score_scale", "") or "")
        threshold_strategy = str(getattr(cfg, "threshold_strategy", "") or "")
        decision_strategy = str(getattr(cfg, "decision_strategy", "") or "")
        genuine_stats = getattr(cfg, "genuine_score_stats", None)
        scaler_path = Path(settings.processed_data_path) / "z-score" / user_id / "scaler.json"

        def _build_details(*, threshold_finite: bool, genuine_band_ok: bool) -> Dict[str, Any]:
            return {
                "user_id": str(user_id),
                "policy_status": policy_status,
                "policy_search_completed": policy_search_completed,
                "policy_source": policy_source,
                "policy_file": policy_file,
                "score_metric": score_metric,
                "score_scale": score_scale,
                "threshold_strategy": threshold_strategy,
                "decision_strategy": decision_strategy,
                "threshold": (float(cfg.threshold) if cfg.threshold is not None else None),
                "threshold_finite": bool(threshold_finite),
                "genuine_band_ok": bool(genuine_band_ok),
                "genuine_score_stats": genuine_stats if isinstance(genuine_stats, dict) else None,
                "artifacts": {
                    "checkpoint": self._artifact_fingerprint(cfg.vqgan_checkpoint),
                    "config": self._artifact_fingerprint(cfg.vqgan_config if cfg.vqgan_config else None),
                    "scaler": self._artifact_fingerprint(scaler_path),
                    "policy": self._artifact_fingerprint(Path(policy_file) if policy_file else None),
                },
            }

        # READINESS GATE.
        ready = (policy_source == "best" and policy_status == "ready" and policy_search_completed)
        if not ready:
            degraded = False
            if allow_fb and policy_source == "training_fallback":
                degraded = True
            elif allow_fb and policy_status in ("ready", "legacy"):
                degraded = True
            if not degraded:
                detail = (
                    f"source={policy_source or 'unknown'}, status={policy_status or 'unknown'}, "
                    f"completed={policy_search_completed}"
                )
                reason = f"policy_not_ready: {detail} (set auth.allow_training_fallback_policy=true to override)"
                return False, reason, _build_details(threshold_finite=False, genuine_band_ok=True)

        # 1. threshold must be finite (reject inf/nan/sys.float_info.max-like values).
        threshold = cfg.threshold
        threshold_finite = (
            threshold is not None
            and math.isfinite(float(threshold))
            and abs(float(threshold)) < 1e6
            and abs(float(threshold)) < sys.float_info.max
        )
        if not threshold_finite:
            return False, f"invalid threshold: {threshold!r}", _build_details(threshold_finite=False, genuine_band_ok=True)

        # 2. required policy fields.
        if not decision_strategy:
            return False, "missing decision_strategy", _build_details(threshold_finite=True, genuine_band_ok=True)
        if not threshold_strategy:
            return False, "missing threshold_strategy", _build_details(threshold_finite=True, genuine_band_ok=True)
        if not score_metric:
            return False, "missing score_metric", _build_details(threshold_finite=True, genuine_band_ok=True)
        if not score_scale:
            return False, "missing score_scale", _build_details(threshold_finite=True, genuine_band_ok=True)

        # 3. artifact existence / integrity.
        if not cfg.vqgan_checkpoint.exists():
            return False, f"missing checkpoint: {cfg.vqgan_checkpoint}", _build_details(threshold_finite=True, genuine_band_ok=True)
        if cfg.vqgan_config and not cfg.vqgan_config.exists():
            return False, f"missing config: {cfg.vqgan_config}", _build_details(threshold_finite=True, genuine_band_ok=True)
        if not scaler_path.exists() or not scaler_path.is_file():
            return False, f"missing scaler: {scaler_path}", _build_details(threshold_finite=True, genuine_band_ok=True)
        try:
            json.loads(scaler_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return False, f"invalid scaler: {exc}", _build_details(threshold_finite=True, genuine_band_ok=True)
        if cfg.vqgan_config:
            try:
                model_cfg = json.loads(cfg.vqgan_config.read_text(encoding="utf-8"))
                if int(model_cfg.get("input_height", 6)) != 6:
                    return (
                        False,
                        f"unsupported model input_height: {model_cfg.get('input_height')}",
                        _build_details(threshold_finite=True, genuine_band_ok=True),
                    )
            except Exception as exc:
                return False, f"invalid config: {exc}", _build_details(threshold_finite=True, genuine_band_ok=True)

        # 4. threshold must lie near the train/val genuine score band (skip when stats absent).
        genuine_band_ok = True
        if isinstance(genuine_stats, dict):
            lo_raw = genuine_stats.get("min")
            hi_raw = genuine_stats.get("max")
            if isinstance(lo_raw, (int, float)) and isinstance(hi_raw, (int, float)):
                lo = float(lo_raw)
                hi = float(hi_raw)
                span = max(hi - lo, 1e-6)
                margin = max(1.0, 0.5 * span)
                if not (lo - margin <= float(threshold) <= hi + margin):
                    return (
                        False,
                        (
                            f"threshold out of genuine score band: threshold={float(threshold):.6f}, "
                            f"band=[{lo:.6f}, {hi:.6f}], margin={margin:.6f}"
                        ),
                        _build_details(threshold_finite=True, genuine_band_ok=False),
                    )

        return True, "", _build_details(threshold_finite=True, genuine_band_ok=genuine_band_ok)

    def check_trained_model(self, user_id: str) -> Tuple[bool, str]:
        ok, reason, _ = self._validate_model(user_id)
        return ok, reason

    def has_trained_model(self, user_id: str) -> bool:
        ready, _ = self.check_trained_model(user_id)
        return ready

    def _write_model_validation(self, user_id: str, session_id: str, details: Dict[str, Any]) -> None:
        """Persist model validation details at session start. Never raises."""
        try:
            import json

            validation_path = (
                Path(settings.inference_storage_path) / user_id / session_id / "model_validation.json"
            )
            validation_path.parent.mkdir(parents=True, exist_ok=True)
            validation_path.write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    def start_session(self, user_id: str, session_id: str) -> Tuple[bool, str, Optional["VQGANPolicy"]]:
        self._prune_sessions()
        from .runner import load_best_policy

        # 认证启动门禁：仅接受通过校验的正式策略（policy_status=ready 且
        # policy_search_completed=true），或显式开启降级开关后的训练兜底策略。
        ok, reason, details = self._validate_model(user_id)
        # 落盘 model_validation.json（无论成功/失败都记录，便于排查）。
        self._write_model_validation(user_id, session_id, details)
        if not ok:
            return False, f"model_not_ready: {reason}", None

        allow_fb = bool(getattr(get_ca_config().auth, "allow_training_fallback_policy", False))
        try:
            cfg = load_best_policy(user_id, models_root=self._models_root, allow_training_fallback=allow_fb)
        except Exception as exc:
            return False, f"model_not_ready: {exc}", None

        policy = self._policy_from_config(cfg)
        state = AuthSessionState(user_id=user_id, session_id=session_id, policy=policy)
        state.consecutive_rejects = ConsecutiveRejectTracker()
        state.vote_rejects = VoteRejectTracker()
        state.ema_rejects = EMAScoreTracker()
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
            "interrupt_rule": str(policy.interrupt_rule),
            "decision_strategy": str(policy.decision_strategy),
            "vote_window_size": int(policy.vote_window_size),
            "vote_min_rejects": int(policy.vote_min_rejects),
            "ema_alpha": float(policy.ema_alpha),
            "model_version": str(policy.model_version),
            "vqgan_checkpoint": str(policy.vqgan_checkpoint),
            "vqgan_config": str(policy.vqgan_config),
        }

    def snapshot_sessions(self) -> List[Dict[str, Any]]:
        self._prune_sessions()
        ca_cfg = get_ca_config()
        sessions: List[Dict[str, Any]] = []
        for state in self._sessions.values():
            consecutive = state.consecutive_rejects
            vote = state.vote_rejects
            ema = state.ema_rejects
            target_windows = windows_for_delay(
                ca_cfg.auth.result_delay_sec,
                window_size_sec=state.policy.window_size,
                overlap=state.policy.overlap,
            )
            sessions.append(
                {
                    "device_id_hash": str(state.user_id),
                    "session_id": str(state.session_id),
                    "created_at": self._iso_from_epoch(state.created_at),
                    "last_activity": self._iso_from_epoch(state.last_activity),
                    "idle_seconds": max(0.0, float(time.time() - state.last_activity)),
                    "window_index": int(state.window_index),
                    # 认证结果发布节流观测：延迟秒数、折算窗口数、当前已累积未发布窗口数。
                    "result_delay_sec": float(ca_cfg.auth.result_delay_sec),
                    "result_emit_windows": int(target_windows),
                    "windows_since_emit": int(getattr(state.emit_gate, "pending", 0)),
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
                    "ema_rejects": {
                        "windows": int(getattr(ema, "windows", 0) or 0),
                        "interrupts": int(getattr(ema, "interrupts", 0) or 0),
                        "first_interrupt_window": getattr(ema, "first_interrupt_window", None),
                        "ema_score": getattr(ema, "ema_score", None),
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

            # 认证结果发布延迟：把 auth.result_delay_sec 折算成需累积的窗口数。
            # 例：window=0.2s、overlap=0.5 => stride=0.1s（每秒约 10 窗），delay=5s => 50 窗。
            # target_windows<=0 表示不延迟（每个数据包都发布最近窗口结果，兼容旧行为）。
            ca_cfg = get_ca_config()
            target_windows = windows_for_delay(
                ca_cfg.auth.result_delay_sec,
                window_size_sec=state.policy.window_size,
                overlap=state.policy.overlap,
            )
            state.emit_gate.target_windows = int(target_windows)

            emit_payload: Optional[AuthResultPayload] = None
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
                ema_score: Optional[float] = None
                vote_recent_windows = 0
                vote_recent_rejects = 0
                decision_strategy = str(state.policy.decision_strategy or state.policy.interrupt_rule).strip().lower()
                if decision_strategy == "ema":
                    interrupt = state.ema_rejects.update(
                        raw_score,
                        alpha=float(state.policy.ema_alpha),
                        threshold=float(state.policy.threshold),
                        reset_on_interrupt=False,
                    )
                    ema_score = None if state.ema_rejects.ema_score is None else float(state.ema_rejects.ema_score)
                    decision_accept = not bool(interrupt)
                    decision_score = float(1.0 / (1.0 + math.exp(-float(ema_score or raw_score))))
                    decision_threshold = float(1.0 / (1.0 + math.exp(-float(state.policy.threshold))))
                    decision_message = (
                        f"EMA={ema_score:.6f}, alpha={state.policy.ema_alpha:.3f}, threshold={state.policy.threshold:.6f}"
                        if ema_score is not None
                        else ""
                    )
                elif state.policy.vote_window_size > 0 and state.policy.vote_min_rejects > 0:
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
                    decision_accept = not bool(interrupt)

                await self._inference_storage.append_result(
                    user_id,
                    session_id,
                    {
                        "window_id": window_id,
                        "score": raw_score,
                        "threshold": float(state.policy.threshold),
                        "raw_accept": accept,
                        "decision_accept": decision_accept,
                        "accept": decision_accept,
                        "interrupt": bool(interrupt),
                        "normalized_score": normalized_score,
                        "decision_strategy": decision_strategy,
                        "decision_score": decision_score,
                        "decision_threshold": decision_threshold,
                        "ema_score": ema_score,
                        "ema_alpha": float(state.policy.ema_alpha),
                        "k_rejects": int(state.policy.k_rejects),
                        "vote_recent_windows": vote_recent_windows,
                        "vote_recent_rejects": vote_recent_rejects,
                        "window_size": float(state.policy.window_size),
                        "model_version": state.policy.model_version,
                        "result_stage": "primary",
                        "raw_score": raw_score,
                        "raw_threshold": float(state.policy.threshold),
                        "ema_threshold": (
                            float(state.policy.threshold) if decision_strategy == "ema" else None
                        ),
                        "display_score": decision_score,
                        "display_threshold": decision_threshold,
                    },
                )

                window_payload = AuthResultPayload(
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
                    result_stage="primary",
                    raw_score=raw_score,
                    raw_threshold=float(state.policy.threshold),
                    ema_score=ema_score,
                    ema_threshold=(
                        float(state.policy.threshold) if decision_strategy == "ema" else None
                    ),
                    display_score=decision_score,
                    display_threshold=decision_threshold,
                )

                # 发布节流：累计满 target_windows 个窗口才把最近一次（已聚合）结果回给 App，
                # 期间数据包只回 Ack。target_windows<=0 时不节流（每个数据包都发布）。
                if target_windows <= 0:
                    emit_payload = window_payload
                elif state.emit_gate.feed():
                    emit_payload = window_payload

            state.window_index += len(window_ids)
            self._trim_tail(state, combined_records)
            return emit_payload

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
