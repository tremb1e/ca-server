import json

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

    def fake_run_window_sweep_for_user(user_id) -> None:
        calls.append(("train", user_id))

    monkeypatch.setattr(training_manager, "process_user", fake_process_user)
    monkeypatch.setattr(training_manager, "run_window_sweep_for_user", fake_run_window_sweep_for_user)

    manager = training_manager.TrainingManager(max_concurrent=1, check_interval_sec=1)
    await manager.submit_if_ready("user1")

    task = manager._tasks.get("user1")
    assert task is not None
    await task

    assert ("process", "user1") in calls
    assert ("train", "user1") in calls

    state_path = tmp_path / "models" / "user1" / "training_state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["status"] == "completed"
