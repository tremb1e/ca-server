from __future__ import annotations

import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..config import settings
from ..training.manager import load_state
from ..utils.path_safety import safe_child_path, validate_storage_id
from ..utils.tls import probe_tls_configuration
from .runtime import RuntimeContext


def _models_root() -> Path:
    return Path(settings.data_storage_path).parent / "models"


def _size_mb(size_bytes: int) -> float:
    return round(float(size_bytes) / (1024 * 1024), 3)


def _iso_from_timestamp(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()


def _file_info(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size_bytes": 0,
            "size_mb": 0.0,
            "modified_at": None,
        }
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": int(st.st_size),
        "size_mb": _size_mb(int(st.st_size)),
        "modified_at": _iso_from_timestamp(st.st_mtime),
    }


def _safe_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


def _iter_dirs(path: Path) -> Iterable[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir()], key=lambda p: p.name)


def _sum_files(paths: Iterable[Path]) -> Dict[str, Any]:
    total_size = 0
    count = 0
    latest_mtime: Optional[float] = None
    for path in paths:
        try:
            st = path.stat()
        except FileNotFoundError:
            continue
        count += 1
        total_size += int(st.st_size)
        latest_mtime = st.st_mtime if latest_mtime is None else max(latest_mtime, st.st_mtime)
    return {
        "file_count": int(count),
        "total_size_bytes": int(total_size),
        "total_size_mb": _size_mb(total_size),
        "latest_modified_at": None if latest_mtime is None else _iso_from_timestamp(latest_mtime),
    }


def get_device_ids() -> List[str]:
    device_ids: set[str] = set()
    roots = [
        Path(settings.data_storage_path),
        _models_root(),
        Path(settings.inference_storage_path),
        Path(settings.processed_data_path) / "z-score",
    ]
    for root in roots:
        for entry in _iter_dirs(root):
            device_ids.add(entry.name)

    window_root = Path(settings.processed_data_path) / "window"
    for window_dir in _iter_dirs(window_root):
        for entry in _iter_dirs(window_dir):
            device_ids.add(entry.name)
    return sorted(device_ids)


def storage_stats(root: Path) -> Dict[str, Any]:
    root = Path(root)
    total_devices = 0
    total_sessions = 0
    total_size = 0
    if root.exists():
        for device_dir in _iter_dirs(root):
            total_devices += 1
            for session_file in device_dir.glob("session_*.jsonl"):
                try:
                    total_sessions += 1
                    total_size += int(session_file.stat().st_size)
                except FileNotFoundError:
                    continue
    return {
        "base_path": str(root),
        "total_devices": int(total_devices),
        "total_sessions": int(total_sessions),
        "total_size_bytes": int(total_size),
        "total_size_mb": _size_mb(total_size),
    }


def raw_summary(device_id: str) -> Dict[str, Any]:
    root = safe_child_path(Path(settings.data_storage_path), validate_storage_id(device_id, field_name="device_id"))
    sessions = list(root.glob("session_*.jsonl")) if root.exists() else []
    summary = _sum_files(sessions)
    return {
        "exists": bool(root.exists()),
        "path": str(root),
        "sessions": int(summary["file_count"]),
        "total_size_bytes": int(summary["total_size_bytes"]),
        "total_size_mb": float(summary["total_size_mb"]),
        "latest_activity_at": summary["latest_modified_at"],
    }


def processed_summary(device_id: str) -> Dict[str, Any]:
    device_id = validate_storage_id(device_id, field_name="device_id")
    processed_root = Path(settings.processed_data_path)
    z_dir = processed_root / "z-score" / device_id
    split_files = {name: _file_info(z_dir / f"{name}.csv") for name in ("train", "val", "test")}
    scaler = _file_info(z_dir / "scaler.json")
    windows: Dict[str, Dict[str, Any]] = {}
    window_root = processed_root / "window"
    for window_dir in _iter_dirs(window_root):
        user_dir = window_dir / device_id
        if not user_dir.exists():
            continue
        windows[window_dir.name] = {
            name: _file_info(user_dir / f"{name}.csv")
            for name in ("train", "val", "test")
        }
    files = [Path(info["path"]) for info in split_files.values()]
    if scaler["exists"]:
        files.append(Path(scaler["path"]))
    for window_payload in windows.values():
        files.extend(Path(info["path"]) for info in window_payload.values())
    total = _sum_files([path for path in files if path.exists()])
    return {
        "exists": z_dir.exists() or bool(windows),
        "z_score_dir": str(z_dir),
        "splits": split_files,
        "scaler": scaler,
        "windows": windows,
        "total_size_bytes": int(total["total_size_bytes"]),
        "total_size_mb": float(total["total_size_mb"]),
        "latest_modified_at": total["latest_modified_at"],
    }


def inference_summary(device_id: str) -> Dict[str, Any]:
    root = safe_child_path(Path(settings.inference_storage_path), validate_storage_id(device_id, field_name="device_id"))
    files = list(root.glob("*/*.jsonl")) if root.exists() else []
    results_files = list(root.glob("*/results.jsonl")) if root.exists() else []
    total = _sum_files(files)
    return {
        "exists": bool(root.exists()),
        "path": str(root),
        "sessions": len([p for p in _iter_dirs(root)]),
        "result_sessions": len(results_files),
        "total_size_bytes": int(total["total_size_bytes"]),
        "total_size_mb": float(total["total_size_mb"]),
        "latest_modified_at": total["latest_modified_at"],
    }


def training_status(device_id: str, ctx: RuntimeContext) -> Dict[str, Any]:
    device_id = validate_storage_id(device_id, field_name="device_id")
    readiness = ctx.training_manager.get_readiness(device_id)
    state = load_state(ctx.models_root, device_id)
    task_snapshot = None
    for task in ctx.training_manager.snapshot_tasks().get("tasks", []):
        if task.get("user_id") == device_id:
            task_snapshot = task
            break
    return {
        "device_id_hash": str(device_id),
        "status": str(readiness.status),
        "total_bytes": int(readiness.total_bytes),
        "total_mb": _size_mb(int(readiness.total_bytes)),
        "min_bytes": int(readiness.min_bytes),
        "min_mb": _size_mb(int(readiness.min_bytes)),
        "has_enough_data": bool(readiness.has_enough_data),
        "is_ready": bool(readiness.is_ready),
        "last_trained_bytes": int(state.last_trained_bytes),
        "last_error": str(state.last_error or readiness.last_error or ""),
        "updated_at": str(state.updated_at or ""),
        "task": task_snapshot,
    }


def model_info(device_id: str) -> Dict[str, Any]:
    device_id = validate_storage_id(device_id, field_name="device_id")
    models_root = _models_root()
    user_dir = safe_child_path(models_root, device_id)
    policy_path = user_dir / "best_lock_policy.json"
    files: Dict[str, Dict[str, Any]] = {"policy": _file_info(policy_path)}
    policy_payload: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    try:
        from ..authentication.runner import load_best_policy

        cfg = load_best_policy(device_id, models_root=models_root, policy_path=policy_path)
        files["vqgan_checkpoint"] = _file_info(cfg.vqgan_checkpoint)
        files["vqgan_config"] = _file_info(cfg.vqgan_config)
        policy_payload = {
            "user": str(cfg.user),
            "window_size": float(cfg.window_size),
            "target_width": int(cfg.target_width),
            "overlap": float(cfg.overlap),
            "threshold": float(cfg.threshold),
            "interrupt_rule": str(cfg.interrupt_rule),
            "k_rejects": int(cfg.k_rejects),
            "vote_window_size": int(cfg.vote_window_size),
            "vote_min_rejects": int(cfg.vote_min_rejects),
            "model_version": str(cfg.model_version or cfg.vqgan_checkpoint.name),
            "vqgan_checkpoint": str(cfg.vqgan_checkpoint),
            "vqgan_config": str(cfg.vqgan_config),
        }
        raw_policy = _safe_json(policy_path, {})
        raw_user_policy = raw_policy.get(device_id, {}) if isinstance(raw_policy, dict) else {}
        if isinstance(raw_user_policy, dict) and raw_user_policy.get("lm_checkpoint"):
            files["lm_checkpoint"] = _file_info(Path(str(raw_user_policy["lm_checkpoint"])))
    except Exception as exc:  # noqa: BLE001
        error = str(exc)

    summary_raw = _safe_json(user_dir / "training_summary.json", [])
    if isinstance(summary_raw, dict):
        training_summary = [v for v in summary_raw.values() if isinstance(v, dict)]
    elif isinstance(summary_raw, list):
        training_summary = [v for v in summary_raw if isinstance(v, dict)]
    else:
        training_summary = []

    search_dir = user_dir / "policy_search"
    policy_search = {
        "path": str(search_dir),
        "exists": bool(search_dir.exists()),
        "files": {
            path.name: _file_info(path)
            for path in sorted(search_dir.glob("*.csv")) + sorted(search_dir.glob("*.json"))
            if path.is_file()
        },
    }

    ready = bool(policy_payload and files.get("vqgan_checkpoint", {}).get("exists") and files.get("vqgan_config", {}).get("exists"))
    return {
        "device_id_hash": str(device_id),
        "ready": ready,
        "policy": policy_payload,
        "files": files,
        "training_summary": training_summary,
        "policy_search": policy_search,
        "error": error,
    }


def device_list_item(device_id: str, ctx: RuntimeContext) -> Dict[str, Any]:
    model = model_info(device_id)
    return {
        "device_id_hash": str(device_id),
        "raw": raw_summary(device_id),
        "processed": processed_summary(device_id),
        "model": {
            "ready": bool(model["ready"]),
            "model_version": ((model.get("policy") or {}).get("model_version") if model.get("policy") else ""),
            "policy_exists": bool((model.get("files") or {}).get("policy", {}).get("exists")),
        },
        "inference": inference_summary(device_id),
        "training": training_status(device_id, ctx),
    }


def device_detail(device_id: str, ctx: RuntimeContext) -> Dict[str, Any]:
    sessions = [s for s in ctx.auth_manager.snapshot_sessions() if s.get("user_id") == device_id]
    return {
        "device_id_hash": str(device_id),
        "raw": raw_summary(device_id),
        "processed": processed_summary(device_id),
        "inference": inference_summary(device_id),
        "training": training_status(device_id, ctx),
        "model": model_info(device_id),
        "active_auth_sessions": sessions,
    }


def _count_lines(path: Path) -> int:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())
    except FileNotFoundError:
        return 0


def raw_sessions(device_id: str, *, limit: int) -> Dict[str, Any]:
    root = safe_child_path(Path(settings.data_storage_path), validate_storage_id(device_id, field_name="device_id"))
    session_paths = sorted(
        root.glob("session_*.jsonl") if root.exists() else [],
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    limit = max(1, min(int(limit), 1000))
    sessions = []
    for path in session_paths[:limit]:
        info = _file_info(path)
        stem = path.stem
        session_id = stem[len("session_") :] if stem.startswith("session_") else stem
        sessions.append(
            {
                "session_id": session_id,
                "packet_count": _count_lines(path),
                **info,
            }
        )
    return {"device_id_hash": str(device_id), "sessions": sessions, "total": len(session_paths)}


def tail_jsonl(path: Path, *, limit: int) -> List[Dict[str, Any]]:
    limit = max(1, min(int(limit), 1000))
    rows: deque[Dict[str, Any]] = deque(maxlen=limit)
    if not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return list(rows)


def auth_results(device_id: str, *, session_id: Optional[str], limit: int) -> Dict[str, Any]:
    device_id = validate_storage_id(device_id, field_name="device_id")
    root = safe_child_path(Path(settings.inference_storage_path), device_id)
    paths: Sequence[Path]
    if session_id:
        session_id = validate_storage_id(session_id, field_name="session_id")
        paths = [root / session_id / "results.jsonl"]
    else:
        paths = sorted(root.glob("*/results.jsonl") if root.exists() else [], key=lambda p: p.stat().st_mtime, reverse=True)

    out: List[Dict[str, Any]] = []
    for path in paths:
        sid = path.parent.name
        for row in tail_jsonl(path, limit=limit):
            row = dict(row)
            row.setdefault("device_id_hash", device_id)
            row.setdefault("session_id", sid)
            out.append(row)
    out = _sort_results(out)[: max(1, min(int(limit), 1000))]
    return {"results": out, "total": len(out)}


def latest_auth_results(*, limit: int) -> Dict[str, Any]:
    root = Path(settings.inference_storage_path)
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*/*/results.jsonl") if root.exists() else [], key=lambda p: p.stat().st_mtime, reverse=True):
        device_id = path.parent.parent.name
        session_id = path.parent.name
        for row in tail_jsonl(path, limit=min(limit, 100)):
            row = dict(row)
            row.setdefault("device_id_hash", device_id)
            row.setdefault("session_id", session_id)
            rows.append(row)
    rows = _sort_results(rows)[: max(1, min(int(limit), 1000))]
    return {"results": rows, "total": len(rows)}


def _sort_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> str:
        return str(row.get("server_written_timestamp") or row.get("timestamp") or "")

    return sorted(rows, key=key, reverse=True)


def runtime_snapshot(ctx: RuntimeContext) -> Dict[str, Any]:
    tls_probe = probe_tls_configuration(
        settings.tls_certfile,
        settings.tls_keyfile,
        settings.tls_ca_certs,
        key_password=settings.tls_keyfile_password,
        allow_encrypted_key=True,
    )
    try:
        from ..utils.accelerator import detect_backend

        backend = {"selected": detect_backend("auto"), "error": ""}
    except Exception as exc:  # noqa: BLE001
        backend = {"selected": "unknown", "error": str(exc)}

    metrics = ctx.metrics.snapshot()
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "started_at": metrics["started_at"],
        "uptime_seconds": metrics["uptime_seconds"],
        "http": {
            "enabled": bool(settings.http_enabled or settings.management_api_enabled),
            "configured_enabled": bool(settings.http_enabled),
            "management_api_enabled": bool(settings.management_api_enabled),
            "host": str(settings.host),
            "port": int(settings.port),
        },
        "grpc": {
            "host": str(settings.grpc_host),
            "port": int(settings.grpc_port),
            "max_message_size": int(settings.grpc_max_message_size),
            "max_concurrent_rpcs": int(settings.grpc_max_concurrent_rpcs),
        },
        "tls": {
            "enabled": bool(tls_probe.is_tls),
            "reason": tls_probe.reason,
            "certfile": str(tls_probe.certfile) if tls_probe.certfile else None,
            "keyfile": str(tls_probe.keyfile) if tls_probe.keyfile else None,
        },
        "backend": backend,
        "metrics": metrics,
        "training_tasks": ctx.training_manager.snapshot_tasks(),
        "model_cache": ctx.auth_manager.snapshot_model_cache(),
    }


def summary_snapshot(ctx: RuntimeContext) -> Dict[str, Any]:
    devices = get_device_ids()
    runtime = ctx.metrics.snapshot()
    models_root = _models_root()
    model_count = len([p for p in _iter_dirs(models_root) if (p / "best_lock_policy.json").exists()])
    active_sessions = len(ctx.auth_manager.snapshot_sessions())
    training_tasks = ctx.training_manager.snapshot_tasks()
    return {
        "status": "ok",
        "runtime": {
            "started_at": runtime["started_at"],
            "uptime_seconds": runtime["uptime_seconds"],
            "app_name": settings.app_name,
            "version": settings.version,
        },
        "storage": {
            "raw": storage_stats(Path(settings.data_storage_path)),
            "processed_root": str(settings.processed_data_path),
            "inference_root": str(settings.inference_storage_path),
            "models_root": str(models_root),
        },
        "counts": {
            "devices": len(devices),
            "models": int(model_count),
            "active_auth_sessions": int(active_sessions),
            "active_training_tasks": int(training_tasks.get("active_tasks", 0)),
        },
        "recent_errors": runtime["recent_errors"],
    }


def client_metrics_snapshot(ctx: RuntimeContext) -> Dict[str, Any]:
    metrics = ctx.metrics.snapshot()
    return {
        "latest_by_device": metrics["latest_client_metrics_by_device"],
        "recent": metrics["recent_client_metrics"],
    }
