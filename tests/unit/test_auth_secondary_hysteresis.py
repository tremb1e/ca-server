from types import SimpleNamespace

import numpy as np
import pytest

from src.authentication.manager import AuthSessionManager, AuthSessionState
from src.config import settings
from src.utils.reject_trackers import ConsecutiveRejectTracker, EMAScoreTracker, VoteRejectTracker
from src.utils.secondary_hysteresis import SecondaryHysteresisConfig, SecondaryHysteresisTracker


class _FakeParam:
    device = "cpu"


class _FakeModel:
    def parameters(self):
        return iter([_FakeParam()])


@pytest.mark.asyncio
async def test_auth_manager_secondary_vote_emits_only_after_ten_primary_results(tmp_path, monkeypatch):
    user_id = "device-a"
    session_id = "auth-session"
    monkeypatch.setattr(settings, "inference_storage_path", tmp_path / "inference")
    monkeypatch.setattr(settings, "processed_data_path", tmp_path / "processed")
    scaler_path = tmp_path / "processed" / "z-score" / user_id / "scaler.json"
    scaler_path.parent.mkdir(parents=True)
    scaler_path.write_text("{}", encoding="utf-8")

    import src.authentication.vqgan_inference as vqgan_inference
    import src.processing.pipeline as pipeline
    import src.processing.scaler as scaler

    monkeypatch.setattr(pipeline, "build_config", lambda: SimpleNamespace(sampling_rate_hz=100))
    monkeypatch.setattr(
        pipeline,
        "_extract_sensor_records",
        lambda packets: {"acc": [{"timestamp": 1000}], "gyr": [], "mag": []},
    )
    monkeypatch.setattr(pipeline, "_resample_records", lambda *args, **kwargs: SimpleNamespace(empty=False))
    monkeypatch.setattr(scaler, "load_scaler", lambda path: {})
    monkeypatch.setattr(scaler, "apply_scaler", lambda df, loaded_scaler: df)
    monkeypatch.setattr(
        vqgan_inference,
        "windowize_dataframe",
        lambda *args, **kwargs: (np.asarray([0]), np.zeros((1, 9, 20), dtype=np.float32)),
    )
    scores = iter([-1.0] * 8 + [1.0] * 2)
    monkeypatch.setattr(
        vqgan_inference,
        "score_windows",
        lambda *args, **kwargs: np.asarray([next(scores)], dtype=np.float32),
    )

    manager = AuthSessionManager(models_root=tmp_path / "models")
    manager._model_cache.get = lambda policy: _FakeModel()  # type: ignore[method-assign]
    policy = SimpleNamespace(
        user=user_id,
        window_size=0.2,
        overlap=0.5,
        target_width=20,
        threshold=0.0,
        interrupt_rule="ema",
        decision_strategy="ema",
        k_rejects=0,
        vqgan_checkpoint=tmp_path / "model.pt",
        vqgan_config=tmp_path / "model.json",
        vote_window_size=0,
        vote_min_rejects=0,
        ema_alpha=1.0,
        model_version="test-model",
    )
    state = AuthSessionState(user_id=user_id, session_id=session_id, policy=policy)
    state.consecutive_rejects = ConsecutiveRejectTracker()
    state.vote_rejects = VoteRejectTracker()
    state.ema_rejects = EMAScoreTracker()
    state.secondary_hysteresis_config = SecondaryHysteresisConfig(
        strategy="vote",
        primary_result_interval_sec=1.0,
        decision_time_sec=10.0,
        vote_window_size=10,
        vote_min_rejects=8,
    )
    state.secondary_hysteresis = SecondaryHysteresisTracker(state.secondary_hysteresis_config)
    manager._sessions[manager._session_key(user_id, session_id)] = state

    for _ in range(9):
        assert await manager.handle_packet(user_id=user_id, session_id=session_id, parsed_batch={}) is None

    result = await manager.handle_packet(user_id=user_id, session_id=session_id, parsed_batch={})

    assert result is not None
    assert result.accept is False
    assert result.interrupt is True
    assert result.score == 0.2
    assert result.threshold == 0.3

    results_path = tmp_path / "inference" / user_id / session_id / "results.jsonl"
    result_lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 11
    assert sum('"result_stage": "primary"' in line for line in result_lines) == 10
    assert sum('"result_stage": "secondary"' in line for line in result_lines) == 1


@pytest.mark.asyncio
async def test_auth_manager_secondary_consumes_multiple_primary_intervals_from_one_packet(tmp_path, monkeypatch):
    user_id = "device-a"
    session_id = "auth-session"
    monkeypatch.setattr(settings, "inference_storage_path", tmp_path / "inference")
    monkeypatch.setattr(settings, "processed_data_path", tmp_path / "processed")
    scaler_path = tmp_path / "processed" / "z-score" / user_id / "scaler.json"
    scaler_path.parent.mkdir(parents=True)
    scaler_path.write_text("{}", encoding="utf-8")

    import src.authentication.vqgan_inference as vqgan_inference
    import src.processing.pipeline as pipeline
    import src.processing.scaler as scaler

    monkeypatch.setattr(pipeline, "build_config", lambda: SimpleNamespace(sampling_rate_hz=100))
    monkeypatch.setattr(
        pipeline,
        "_extract_sensor_records",
        lambda packets: {"acc": [{"timestamp": 1000}], "gyr": [], "mag": []},
    )
    monkeypatch.setattr(pipeline, "_resample_records", lambda *args, **kwargs: SimpleNamespace(empty=False))
    monkeypatch.setattr(scaler, "load_scaler", lambda path: {})
    monkeypatch.setattr(scaler, "apply_scaler", lambda df, loaded_scaler: df)
    monkeypatch.setattr(
        vqgan_inference,
        "windowize_dataframe",
        lambda *args, **kwargs: (np.arange(19), np.zeros((19, 1, 9, 20), dtype=np.float32)),
    )
    monkeypatch.setattr(
        vqgan_inference,
        "score_windows",
        lambda *args, **kwargs: np.full((19,), -1.0, dtype=np.float32),
    )

    manager = AuthSessionManager(models_root=tmp_path / "models")
    manager._model_cache.get = lambda policy: _FakeModel()  # type: ignore[method-assign]
    policy = SimpleNamespace(
        user=user_id,
        window_size=0.2,
        overlap=0.5,
        target_width=20,
        threshold=0.0,
        interrupt_rule="ema",
        decision_strategy="ema",
        k_rejects=0,
        vqgan_checkpoint=tmp_path / "model.pt",
        vqgan_config=tmp_path / "model.json",
        vote_window_size=0,
        vote_min_rejects=0,
        ema_alpha=1.0,
        model_version="test-model",
    )
    state = AuthSessionState(user_id=user_id, session_id=session_id, policy=policy)
    state.consecutive_rejects = ConsecutiveRejectTracker()
    state.vote_rejects = VoteRejectTracker()
    state.ema_rejects = EMAScoreTracker()
    state.secondary_hysteresis_config = SecondaryHysteresisConfig(
        strategy="vote",
        primary_result_interval_sec=1.0,
        decision_time_sec=2.0,
        vote_window_size=2,
        vote_min_rejects=2,
    )
    state.secondary_hysteresis = SecondaryHysteresisTracker(state.secondary_hysteresis_config)
    manager._sessions[manager._session_key(user_id, session_id)] = state

    result = await manager.handle_packet(user_id=user_id, session_id=session_id, parsed_batch={})

    assert result is not None
    assert result.accept is False
    assert result.interrupt is True
    assert result.window_id == 18
    assert state.secondary_hysteresis.inputs_seen == 2

    results_path = tmp_path / "inference" / user_id / session_id / "results.jsonl"
    result_lines = results_path.read_text(encoding="utf-8").splitlines()
    assert len(result_lines) == 20
    assert sum('"result_stage": "primary"' in line for line in result_lines) == 19
    assert sum('"result_stage": "secondary"' in line for line in result_lines) == 1
    assert '"secondary_sample_every": 10' in result_lines[-1]
    assert '"secondary_sample_phase": 8' in result_lines[-1]
