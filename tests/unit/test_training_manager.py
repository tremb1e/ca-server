import json
import sys
import time
import types

import pytest

from src.ca_config import AuthConfig, CAConfig, ProcessingConfig, WindowConfig
from src.training import manager as training_manager


@pytest.mark.asyncio
async def test_training_manager_triggers_when_threshold_met(tmp_path, monkeypatch) -> None:
    raw_root = tmp_path / "raw_data"
    user_dir = raw_root / "user1"
    user_dir.mkdir(parents=True)
    session_path = user_dir / "session_test.jsonl"
    session_path.write_bytes(b"x" * 2 * 1024 * 1024)

    monkeypatch.setattr(training_manager.settings, "data_storage_path", raw_root)
    monkeypatch.setattr(training_manager.settings, "processed_data_path", tmp_path / "processed_data")

    ca_cfg = CAConfig(
        processing=ProcessingConfig(min_total_mb=1, target_total_mb=1, workers=1),
        windows=WindowConfig(sizes=[0.2], overlap=0.5, sampling_rate_hz=100),
        auth=AuthConfig(max_decision_time_sec=2.0),
    )
    monkeypatch.setattr(training_manager, "get_ca_config", lambda: ca_cfg)

    calls = []

    def fake_process_user(user_id, cfg) -> None:
        calls.append(("process", user_id))

    def fake_run_window_sweep_for_user(user_id, **kwargs) -> None:
        calls.append(("train", user_id, kwargs.get("device")))

    fake_pipeline = types.ModuleType("src.processing.pipeline")
    fake_pipeline.build_config = lambda: object()
    fake_pipeline.process_user = fake_process_user
    fake_runner = types.ModuleType("src.training.runner")
    fake_runner.run_window_sweep_for_user = fake_run_window_sweep_for_user
    monkeypatch.setitem(sys.modules, "src.processing.pipeline", fake_pipeline)
    monkeypatch.setitem(sys.modules, "src.training.runner", fake_runner)

    manager = training_manager.TrainingManager(max_concurrent=1, check_interval_sec=1)
    await manager.submit_if_ready("user1")

    task = manager._tasks.get("user1")
    assert task is not None
    await task

    assert ("process", "user1") in calls
    train_calls = [call for call in calls if call[0] == "train" and call[1] == "user1"]
    assert train_calls
    assert str(train_calls[0][2]).split(":", 1)[0] in {"cpu", "npu", "cuda"}

    state_path = tmp_path / "models" / "user1" / "training_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"


def _install_fake_training_pipeline(monkeypatch, calls) -> None:
    """Stub the lazily-imported training pipeline so submit_if_ready runs the real
    scheduling logic without touching the processing/NPU code."""
    fake_pipeline = types.ModuleType("src.processing.pipeline")
    fake_pipeline.build_config = lambda: object()
    fake_pipeline.process_user = lambda user_id, cfg: calls.append(("process", user_id))
    fake_runner = types.ModuleType("src.training.runner")
    fake_runner.run_window_sweep_for_user = lambda user_id, **kwargs: calls.append(("train", user_id))
    monkeypatch.setitem(sys.modules, "src.processing.pipeline", fake_pipeline)
    monkeypatch.setitem(sys.modules, "src.training.runner", fake_runner)


def _make_ca_cfg() -> CAConfig:
    return CAConfig(
        processing=ProcessingConfig(min_total_mb=1, target_total_mb=1, workers=1),
        windows=WindowConfig(sizes=[0.2], overlap=0.5, sampling_rate_hz=100),
        auth=AuthConfig(max_decision_time_sec=2.0),
    )


@pytest.mark.asyncio
async def test_forced_submit_bypasses_throttle(tmp_path, monkeypatch) -> None:
    """开始认证 issues submit_if_ready(force=True). It must schedule training even
    when a freshly-streamed sensor packet just refreshed the throttle window;
    otherwise the app is told "training_in_progress" while nothing ever trains."""
    raw_root = tmp_path / "raw_data"
    user_dir = raw_root / "user1"
    user_dir.mkdir(parents=True)
    (user_dir / "session_test.jsonl").write_bytes(b"x" * 2 * 1024 * 1024)

    monkeypatch.setattr(training_manager.settings, "data_storage_path", raw_root)
    monkeypatch.setattr(training_manager.settings, "processed_data_path", tmp_path / "processed_data")
    monkeypatch.setattr(training_manager, "get_ca_config", _make_ca_cfg)

    calls = []
    _install_fake_training_pipeline(monkeypatch, calls)

    # Large throttle window, and a sensor packet "just" refreshed _last_checked:
    # the passive path is firmly inside the throttle window.
    manager = training_manager.TrainingManager(max_concurrent=1, check_interval_sec=10_000)
    manager._last_checked["user1"] = time.time()

    # Passive (per-packet) path stays correctly throttled -> no training task.
    await manager.submit_if_ready("user1")
    assert manager._tasks.get("user1") is None

    # Forced (开始认证) path must bypass the throttle and schedule training.
    await manager.submit_if_ready("user1", force=True)
    task = manager._tasks.get("user1")
    assert task is not None
    await task
    assert ("process", "user1") in calls


@pytest.mark.asyncio
async def test_forced_submit_recovers_stale_in_progress(tmp_path, monkeypatch) -> None:
    """A crash/restart can leave training_state.json at status=in_progress with no
    live task. The passive path keeps waiting, but a forced trigger must recover
    instead of being stuck on "training_in_progress" forever."""
    raw_root = tmp_path / "raw_data"
    user_dir = raw_root / "user1"
    user_dir.mkdir(parents=True)
    (user_dir / "session_test.jsonl").write_bytes(b"x" * 2 * 1024 * 1024)

    monkeypatch.setattr(training_manager.settings, "data_storage_path", raw_root)
    monkeypatch.setattr(training_manager.settings, "processed_data_path", tmp_path / "processed_data")
    monkeypatch.setattr(training_manager, "get_ca_config", _make_ca_cfg)

    calls = []
    _install_fake_training_pipeline(monkeypatch, calls)

    models_root = tmp_path / "models"
    # Simulate an interrupted run: persisted in_progress, but no live task.
    training_manager.save_state(
        models_root,
        "user1",
        training_manager.TrainingState(status="in_progress", updated_at="2026-01-01T00:00:00Z"),
    )

    manager = training_manager.TrainingManager(max_concurrent=1, check_interval_sec=1)

    # Passive path must not disturb a possibly-live in_progress run.
    await manager.submit_if_ready("user1")
    assert manager._tasks.get("user1") is None

    # Forced trigger recovers the orphaned state and retrains.
    await manager.submit_if_ready("user1", force=True)
    task = manager._tasks.get("user1")
    assert task is not None
    await task
    assert ("process", "user1") in calls
    payload = json.loads((models_root / "user1" / "training_state.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
