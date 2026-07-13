"""Task 2 — hardened model validation (check_trained_model / _validate_model)
and model_validation.json at session start. (No secondary-hysteresis content.)"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.authentication import manager as auth_manager_mod
from src.authentication.manager import AuthSessionManager

READY = {
    "policy_status": "ready",
    "policy_search_completed": True,
    "score_metric": "mse",
    "score_scale": "negative_reconstruction_error",
    "threshold_strategy": "interrupt_window_frr",
}


def _seed(
    models_root: Path,
    processed_root: Path,
    user: str,
    *,
    extra: dict,
    threshold=-0.12,
    input_height=6,
    filename: str = "best_lock_policy.json",
    genuine_stats=None,
) -> None:
    user_dir = models_root / user
    (user_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (user_dir / "checkpoints" / "vqgan.pt").write_bytes(b"model")
    (user_dir / "checkpoints" / "vqgan.json").write_text(
        json.dumps({"input_height": input_height, "input_width": 20}), encoding="utf-8"
    )
    policy = {
        "user": user,
        "window": 0.2,
        "overlap": 0.5,
        "target_width": 20,
        "threshold": threshold,
        "interrupt_rule": "ema",
        "decision_strategy": "ema",
        "ema_alpha": 0.25,
        "vqgan_checkpoint": "checkpoints/vqgan.pt",
        "vqgan_config": "checkpoints/vqgan.json",
        "model_version": "vqgan.pt",
    }
    policy.update(extra)
    if genuine_stats is not None:
        policy["genuine_score_stats"] = genuine_stats
    (user_dir / filename).write_text(json.dumps({user: policy}), encoding="utf-8")
    z = processed_root / "z-score" / user
    z.mkdir(parents=True, exist_ok=True)
    (z / "scaler.json").write_text("{}", encoding="utf-8")


def _mgr(tmp_path, monkeypatch, *, allow_fb: bool = False):
    models_root = tmp_path / "models"
    processed_root = tmp_path / "processed"
    inference_root = tmp_path / "inference"
    for p in (models_root, processed_root, inference_root):
        p.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(auth_manager_mod.settings, "processed_data_path", processed_root)
    monkeypatch.setattr(auth_manager_mod.settings, "inference_storage_path", inference_root)
    monkeypatch.setattr(auth_manager_mod.settings, "data_storage_path", tmp_path / "raw")
    ca_cfg = SimpleNamespace(
        auth=SimpleNamespace(
            allow_training_fallback_policy=allow_fb,
            decision_strategy="ema",
            vote_window_size=7,
            vote_min_rejects=6,
            ema_alpha=0.25,
        )
    )
    monkeypatch.setattr(auth_manager_mod, "get_ca_config", lambda: ca_cfg)
    mgr = AuthSessionManager(models_root=models_root)
    return mgr, models_root, processed_root, inference_root


def test_ready_best_policy_validates(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch)
    _seed(models_root, processed_root, "u1", extra=READY)
    ok, reason = mgr.check_trained_model("u1")
    assert ok is True
    assert reason == ""


def test_bare_legacy_rejected_when_switch_off(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch, allow_fb=False)
    _seed(models_root, processed_root, "u1", extra={})
    ok, reason = mgr.check_trained_model("u1")
    assert ok is False
    assert reason.startswith("policy_not_ready")


def test_training_fallback_rejected_without_switch(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch, allow_fb=False)
    _seed(
        models_root,
        processed_root,
        "u1",
        extra={
            "policy_status": "training_fallback",
            "policy_search_completed": False,
            "score_metric": "mse",
            "score_scale": "negative_reconstruction_error",
            "threshold_strategy": "training_val_genuine_threshold",
        },
        filename="training_fallback_policy.json",
    )
    ok, _ = mgr.check_trained_model("u1")
    assert ok is False


def test_training_fallback_accepted_with_switch(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch, allow_fb=True)
    _seed(
        models_root,
        processed_root,
        "u1",
        extra={
            "policy_status": "training_fallback",
            "policy_search_completed": False,
            "score_metric": "mse",
            "score_scale": "negative_reconstruction_error",
            "threshold_strategy": "training_val_genuine_threshold",
        },
        filename="training_fallback_policy.json",
    )
    ok, reason = mgr.check_trained_model("u1")
    assert ok is True, reason


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan"), sys.float_info.max, 1e7])
def test_non_finite_threshold_rejected(tmp_path, monkeypatch, bad) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch)
    _seed(models_root, processed_root, "u1", extra=READY, threshold=bad)
    ok, reason = mgr.check_trained_model("u1")
    assert ok is False
    assert "threshold" in reason


def test_missing_threshold_strategy_rejected(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch)
    extra = dict(READY)
    extra["threshold_strategy"] = ""
    _seed(models_root, processed_root, "u1", extra=extra)
    ok, reason = mgr.check_trained_model("u1")
    assert ok is False
    assert "threshold_strategy" in reason


def test_wrong_input_height_rejected(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch)
    _seed(models_root, processed_root, "u1", extra=READY, input_height=12)
    ok, reason = mgr.check_trained_model("u1")
    assert ok is False
    assert "input_height" in reason


def test_healthy_nine_axis_model_accepted(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch)
    _seed(models_root, processed_root, "u1", extra=READY, input_height=9)
    ok, reason = mgr.check_trained_model("u1")
    assert ok is True, reason


def test_genuine_band_rejects_far_threshold(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch)
    stats = {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 0.5, "count": 100, "quantiles": {}}
    _seed(models_root, processed_root, "u1", extra=READY, threshold=50.0, genuine_stats=stats)
    ok, reason = mgr.check_trained_model("u1")
    assert ok is False
    assert "genuine" in reason


def test_genuine_band_accepts_in_band_threshold(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch)
    stats = {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 0.5, "count": 100, "quantiles": {}}
    _seed(models_root, processed_root, "u1", extra=READY, threshold=-0.5, genuine_stats=stats)
    ok, reason = mgr.check_trained_model("u1")
    assert ok is True, reason


def test_start_session_writes_model_validation(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, inference_root = _mgr(tmp_path, monkeypatch)
    _seed(models_root, processed_root, "u1", extra=READY)
    ok, message, policy = mgr.start_session("u1", "sess-a")
    assert ok is True, message
    assert policy is not None
    vpath = inference_root / "u1" / "sess-a" / "model_validation.json"
    assert vpath.exists()
    details = json.loads(vpath.read_text(encoding="utf-8"))
    assert details["policy_status"] == "ready"
    assert details["policy_source"] == "best"
    assert set(details["artifacts"].keys()) == {"checkpoint", "config", "scaler", "policy"}
    assert details["artifacts"]["checkpoint"]["exists"] is True


def test_start_session_rejected_when_not_ready(tmp_path, monkeypatch) -> None:
    mgr, models_root, processed_root, _ = _mgr(tmp_path, monkeypatch, allow_fb=False)
    _seed(models_root, processed_root, "u1", extra={})  # bare legacy -> not ready
    ok, message, policy = mgr.start_session("u1", "sess-a")
    assert ok is False
    assert policy is None
    assert message.startswith("model_not_ready")
