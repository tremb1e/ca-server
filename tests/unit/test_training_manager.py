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


def _install_fake_pipeline(monkeypatch, calls, *, on_train=None) -> None:
    def fake_process_user(user_id, cfg) -> None:
        calls.append(("process", user_id))

    def fake_run_window_sweep_for_user(user_id, **kwargs) -> None:
        calls.append(("train", user_id))
        if on_train is not None:
            on_train(user_id, **kwargs)

    fake_pipeline = types.ModuleType("src.processing.pipeline")
    fake_pipeline.build_config = lambda: object()
    fake_pipeline.process_user = fake_process_user
    fake_runner = types.ModuleType("src.training.runner")
    fake_runner.run_window_sweep_for_user = fake_run_window_sweep_for_user
    monkeypatch.setitem(sys.modules, "src.processing.pipeline", fake_pipeline)
    monkeypatch.setitem(sys.modules, "src.training.runner", fake_runner)


def _seed_user_raw(raw_root, user: str = "user1", *, mb: int = 2) -> None:
    user_dir = raw_root / user
    user_dir.mkdir(parents=True, exist_ok=True)
    (user_dir / "session_test.jsonl").write_bytes(b"x" * mb * 1024 * 1024)


def _ca_cfg() -> CAConfig:
    return CAConfig(
        processing=ProcessingConfig(min_total_mb=1, target_total_mb=1, workers=1),
        windows=WindowConfig(sizes=[0.2], overlap=0.5, sampling_rate_hz=100),
        auth=AuthConfig(max_decision_time_sec=2.0),
    )


@pytest.mark.asyncio
async def test_forced_submit_bypasses_throttle(tmp_path, monkeypatch) -> None:
    raw_root = tmp_path / "raw_data"
    _seed_user_raw(raw_root)
    monkeypatch.setattr(training_manager.settings, "data_storage_path", raw_root)
    monkeypatch.setattr(training_manager, "get_ca_config", _ca_cfg)
    calls: list = []
    _install_fake_pipeline(monkeypatch, calls)

    manager = training_manager.TrainingManager(max_concurrent=1, check_interval_sec=10_000)
    manager._last_checked["user1"] = time.time()  # the passive path would be throttled

    await manager.submit_if_ready("user1")  # passive: throttled -> nothing scheduled
    assert manager._tasks.get("user1") is None

    await manager.submit_if_ready("user1", force=True)  # forced: bypass throttle
    task = manager._tasks.get("user1")
    assert task is not None
    await task
    assert ("process", "user1") in calls


@pytest.mark.asyncio
async def test_forced_submit_recovers_stale_in_progress(tmp_path, monkeypatch) -> None:
    raw_root = tmp_path / "raw_data"
    _seed_user_raw(raw_root)
    monkeypatch.setattr(training_manager.settings, "data_storage_path", raw_root)
    monkeypatch.setattr(training_manager, "get_ca_config", _ca_cfg)
    calls: list = []
    _install_fake_pipeline(monkeypatch, calls)

    models_root = raw_root.parent / "models"
    # Persisted in_progress with no live task = interrupted run (the deadlock case).
    training_manager.save_state(models_root, "user1", training_manager.TrainingState(status="in_progress"))

    manager = training_manager.TrainingManager(max_concurrent=1, check_interval_sec=1)

    await manager.submit_if_ready("user1")  # passive: stale in_progress -> waits
    assert manager._tasks.get("user1") is None

    await manager.submit_if_ready("user1", force=True)  # forced: recovers
    task = manager._tasks.get("user1")
    assert task is not None
    await task
    payload = json.loads((models_root / "user1" / "training_state.json").read_text(encoding="utf-8"))
    assert payload["status"] == "completed"


@pytest.mark.asyncio
async def test_run_training_marks_failed_when_policy_search_incomplete(tmp_path, monkeypatch) -> None:
    from src.training.runner import PolicySearchIncompleteError

    raw_root = tmp_path / "raw_data"
    _seed_user_raw(raw_root)
    monkeypatch.setattr(training_manager.settings, "data_storage_path", raw_root)
    monkeypatch.setattr(training_manager, "get_ca_config", _ca_cfg)

    def boom(user_id, **kwargs):
        raise PolicySearchIncompleteError(
            "Training finished but produced no auth-accepted best policy for user=user1"
        )

    calls: list = []
    _install_fake_pipeline(monkeypatch, calls, on_train=boom)

    manager = training_manager.TrainingManager(max_concurrent=1, check_interval_sec=1)
    await manager.submit_if_ready("user1", force=True)
    task = manager._tasks.get("user1")
    assert task is not None
    await task

    models_root = raw_root.parent / "models"
    payload = json.loads((models_root / "user1" / "training_state.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "best policy" in payload["last_error"]
