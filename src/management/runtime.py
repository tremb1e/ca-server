from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Optional

from ..authentication.manager import AuthSessionManager
from ..config import settings
from ..storage.file_storage import FileStorage
from ..training.manager import TrainingManager


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat()


@dataclass
class RuntimeMetrics:
    started_at: datetime = field(default_factory=utc_now)
    counters: Dict[str, int] = field(default_factory=dict)
    recent_errors: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))
    recent_auth_results: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=500))
    recent_client_metrics: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=200))
    latest_client_metrics_by_device: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_packet_by_device: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_auth_result_by_device: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    inference_latency_total_ms: int = 0
    inference_latency_max_ms: int = 0

    def inc(self, name: str, value: int = 1) -> None:
        self.counters[name] = int(self.counters.get(name, 0)) + int(value)

    def _record_error(self, *, device_id: str, reason: str, detail: Optional[str]) -> None:
        if not reason:
            return
        self.recent_errors.appendleft(
            {
                "timestamp": iso_now(),
                "device_id_hash": str(device_id),
                "reason": str(reason),
                "detail": str(detail or ""),
            }
        )

    def record_packet(
        self,
        *,
        device_id: str,
        session_id: str,
        packet_id: str,
        packet_seq_no: int,
        status: str,
        storage_ok: bool,
        error_detail: Optional[str],
    ) -> None:
        self.inc("packets_received")
        self.inc(f"packet_status_{status}")
        if status == "parsed_sensor_batch":
            self.inc("packets_parsed")
        if storage_ok:
            self.inc("packets_stored")
        else:
            self.inc("packets_not_stored")
            if status == "storage_failed":
                self.inc("packet_storage_failed")
        if status in {
            "decrypt_failed",
            "decompress_failed",
            "parse_failed",
            "validation_failed",
            "invalid_identifier",
            "request_too_large",
            "no_data",
            "storage_failed",
            "internal_error",
            "skipped",
        }:
            self._record_error(device_id=device_id, reason=status, detail=error_detail)

        self.last_packet_by_device[str(device_id)] = {
            "timestamp": iso_now(),
            "device_id_hash": str(device_id),
            "session_id": str(session_id),
            "packet_id": str(packet_id),
            "packet_seq_no": int(packet_seq_no),
            "status": str(status),
            "storage_ok": bool(storage_ok),
        }

    def record_auth_result(self, payload: Any, *, inference_latency_ms: int) -> None:
        self.inc("auth_results")
        if getattr(payload, "accept", False):
            self.inc("auth_accepts")
        else:
            self.inc("auth_rejects")
        if getattr(payload, "interrupt", False):
            self.inc("auth_interrupts")
        latency = max(0, int(inference_latency_ms))
        self.inference_latency_total_ms += latency
        self.inference_latency_max_ms = max(self.inference_latency_max_ms, latency)
        record = {
            "timestamp": iso_now(),
            "device_id_hash": str(getattr(payload, "user", "")),
            "session_id": str(getattr(payload, "session_id", "")),
            "window_id": int(getattr(payload, "window_id", 0)),
            "score": float(getattr(payload, "score", 0.0)),
            "threshold": float(getattr(payload, "threshold", 0.0)),
            "accept": bool(getattr(payload, "accept", False)),
            "interrupt": bool(getattr(payload, "interrupt", False)),
            "normalized_score": float(getattr(payload, "normalized_score", 0.0)),
            "k_rejects": int(getattr(payload, "k_rejects", 0)),
            "window_size": float(getattr(payload, "window_size", 0.0)),
            "model_version": str(getattr(payload, "model_version", "")),
            "message": str(getattr(payload, "message", "")),
            "inference_latency_ms": latency,
        }
        self.recent_auth_results.appendleft(record)
        self.last_auth_result_by_device[record["device_id_hash"]] = record

    def record_client_metrics(self, payload: Dict[str, Any]) -> None:
        device_id = str(payload.get("device_id_hash", "") or "unknown_device")
        record = dict(payload)
        record["received_at"] = iso_now()
        self.recent_client_metrics.appendleft(record)
        self.latest_client_metrics_by_device[device_id] = record
        self.inc("client_metric_reports")

    def snapshot(self) -> Dict[str, Any]:
        auth_count = int(self.counters.get("auth_results", 0))
        avg_latency = 0.0
        if auth_count:
            avg_latency = float(self.inference_latency_total_ms / auth_count)
        return {
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": max(0.0, (utc_now() - self.started_at).total_seconds()),
            "counters": dict(self.counters),
            "inference_latency": {
                "avg_ms": avg_latency,
                "max_ms": int(self.inference_latency_max_ms),
            },
            "last_packet_by_device": dict(self.last_packet_by_device),
            "last_auth_result_by_device": dict(self.last_auth_result_by_device),
            "recent_errors": list(self.recent_errors),
            "recent_auth_results": list(self.recent_auth_results),
            "recent_client_metrics": list(self.recent_client_metrics),
            "latest_client_metrics_by_device": dict(self.latest_client_metrics_by_device),
        }


@dataclass
class RuntimeContext:
    storage: FileStorage
    training_manager: TrainingManager
    auth_manager: AuthSessionManager
    metrics: RuntimeMetrics
    models_root: Path


_RUNTIME_CONTEXT: Optional[RuntimeContext] = None


def create_runtime_context() -> RuntimeContext:
    storage = FileStorage(settings.data_storage_path)
    models_root = Path(settings.data_storage_path).parent / "models"
    return RuntimeContext(
        storage=storage,
        training_manager=TrainingManager(
            max_concurrent=settings.training_max_concurrent,
            check_interval_sec=settings.training_check_interval_sec,
        ),
        auth_manager=AuthSessionManager(
            max_cached_models=settings.auth_max_cached_models,
            session_ttl_sec=settings.auth_session_ttl_sec,
            max_concurrent_inference=settings.auth_max_concurrent,
            models_root=models_root,
        ),
        metrics=RuntimeMetrics(),
        models_root=models_root,
    )


def get_runtime_context() -> RuntimeContext:
    global _RUNTIME_CONTEXT
    if _RUNTIME_CONTEXT is None:
        _RUNTIME_CONTEXT = create_runtime_context()
    return _RUNTIME_CONTEXT


def reset_runtime_context() -> RuntimeContext:
    global _RUNTIME_CONTEXT
    _RUNTIME_CONTEXT = create_runtime_context()
    return _RUNTIME_CONTEXT
