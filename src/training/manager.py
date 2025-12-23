from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..ca_config import get_ca_config
from ..config import settings
from ..processing.pipeline import build_config, process_user
from .runner import run_window_sweep_for_user

logger = logging.getLogger(__name__)


@dataclass
class TrainingState:
    status: str = "pending"
    last_trained_bytes: int = 0
    last_error: str = ""
    updated_at: str = ""


@dataclass
class TrainingReadiness:
    status: str
    total_bytes: int
    min_bytes: int
    last_error: str = ""

    @property
    def has_enough_data(self) -> bool:
        return int(self.total_bytes) >= int(self.min_bytes)

    @property
    def is_ready(self) -> bool:
        return self.status == "completed" and self.has_enough_data


def _state_path(models_root: Path, user_id: str) -> Path:
    return models_root / user_id / "training_state.json"


def load_state(models_root: Path, user_id: str) -> TrainingState:
    path = _state_path(models_root, user_id)
    if not path.exists():
        return TrainingState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return TrainingState()
    if not isinstance(payload, dict):
        return TrainingState()
    return TrainingState(
        status=str(payload.get("status", "pending")),
        last_trained_bytes=int(payload.get("last_trained_bytes", 0) or 0),
        last_error=str(payload.get("last_error", "") or ""),
        updated_at=str(payload.get("updated_at", "") or ""),
    )


def save_state(models_root: Path, user_id: str, state: TrainingState) -> None:
    path = _state_path(models_root, user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": state.status,
        "last_trained_bytes": int(state.last_trained_bytes),
        "last_error": state.last_error,
        "updated_at": state.updated_at,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _user_total_bytes(raw_root: Path, user_id: str) -> int:
    user_dir = raw_root / user_id
    if not user_dir.exists():
        return 0
    total = 0
    for session in user_dir.glob("*.jsonl"):
        try:
            total += int(session.stat().st_size)
        except FileNotFoundError:
            continue
    return total


class TrainingManager:
    def __init__(self, *, max_concurrent: int = 1, check_interval_sec: int = 30) -> None:
        self._models_root = Path(settings.data_storage_path).parent / "models"
        self._raw_root = Path(settings.data_storage_path)
        max_concurrent = int(max_concurrent)
        if max_concurrent <= 0:
            max_concurrent = 1
        self._check_interval_sec = max(1, int(check_interval_sec))
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_checked: dict[str, float] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def get_readiness(self, user_id: str) -> TrainingReadiness:
        ca_cfg = get_ca_config()
        total_bytes = _user_total_bytes(self._raw_root, user_id)
        min_bytes = int(float(ca_cfg.processing.min_total_mb) * 1024 * 1024)
        state = load_state(self._models_root, user_id)
        return TrainingReadiness(
            status=state.status,
            total_bytes=int(total_bytes),
            min_bytes=int(min_bytes),
            last_error=str(state.last_error or ""),
        )

    async def submit_if_ready(self, user_id: str, *, force: bool = False) -> None:
        now = time.time()
        last = self._last_checked.get(user_id, 0.0)
        if (now - last) < self._check_interval_sec:
            return
        self._last_checked[user_id] = now

        state = load_state(self._models_root, user_id)
        if state.status == "in_progress":
            return
        if user_id in self._tasks and not self._tasks[user_id].done():
            return

        raw_total = _user_total_bytes(self._raw_root, user_id)
        ca_cfg = get_ca_config()
        if raw_total < int(ca_cfg.processing.min_total_mb * 1024 * 1024):
            return
        if state.status == "completed" and not force:
            return

        logger.info("Training trigger: user=%s total_bytes=%d", user_id, raw_total)
        self._tasks[user_id] = asyncio.create_task(self._run_training(user_id, raw_total))

    async def _run_training(self, user_id: str, raw_total: int) -> None:
        async with self._semaphore:
            state = load_state(self._models_root, user_id)
            state.status = "in_progress"
            state.last_error = ""
            state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            save_state(self._models_root, user_id, state)

            try:
                proc_cfg = build_config()
                await asyncio.to_thread(process_user, user_id, proc_cfg)
                await asyncio.to_thread(run_window_sweep_for_user, user_id)
                state.status = "completed"
                state.last_trained_bytes = int(raw_total)
                state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                save_state(self._models_root, user_id, state)
                logger.info("Training completed: user=%s", user_id)
            except Exception as exc:
                state.status = "failed"
                state.last_error = str(exc)
                state.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                save_state(self._models_root, user_id, state)
                logger.exception("Training failed: user=%s error=%s", user_id, exc)
