"""Functional coverage for check_trained_model gating + validation (contract E).

We build a real AuthSessionManager(models_root=tmp), point settings at tmp dirs,
seed scaler/checkpoint/config, and seed policy json variants on disk.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.authentication.manager import AuthSessionManager
from src.config import settings


def _seed_artifacts(tmp_path: Path, monkeypatch, user: str) -> Path:
    """Point settings at tmp, seed scaler + checkpoint + config(input_height=9).

    Returns the models_root.
    """
    processed_root = tmp_path / "processed"
    inference_root = tmp_path / "inference"
    models_root = tmp_path / "models"
    processed_root.mkdir(parents=True, exist_ok=True)
    inference_root.mkdir(parents=True, exist_ok=True)
    models_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(settings, "processed_data_path", processed_root)
    monkeypatch.setattr(settings, "inference_storage_path", inference_root)

    scaler_path = processed_root / "z-score" / user / "scaler.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    scaler_path.write_text("{}", encoding="utf-8")

    ckpt = models_root / user / "checkpoints" / "vqgan.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"model-bytes")
    cfg = ckpt.with_suffix(".json")
    cfg.write_text(
        json.dumps({"base_channels": 16, "latent_dim": 32, "input_height": 9, "input_width": 20}),
        encoding="utf-8",
    )
    return models_root


def _write_policy(models_root: Path, user: str, *, name: str, extra: dict) -> Path:
    base = {
        "user": user,
        "window": 0.2,
        "overlap": 0.5,
        "target_width": 20,
        "threshold": -0.12,
        "interrupt_rule": "k",
        "decision_strategy": "k",
        "k_rejects": 20,
        "vote_window_size": 0,
        "vote_min_rejects": 0,
        "ema_alpha": 0.25,
        "vqgan_checkpoint": "checkpoints/vqgan.pt",
        "vqgan_config": "checkpoints/vqgan.json",
        "model_version": "vqgan.pt",
    }
    base.update(extra)
    path = models_root / user / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({user: base}), encoding="utf-8")
    return path


def _ready_extra() -> dict:
    return {
        "policy_status": "ready",
        "policy_search_completed": True,
        "threshold_strategy": "interrupt_window_frr",
        "score_metric": "mse",
        "score_scale": "negative_reconstruction_error",
    }


def _patch_ca_config(monkeypatch, *, allow_fb: bool) -> None:
    """Override the (cached) ca_config seen by the manager.

    get_ca_config() is cached in src.ca_config._CACHED, so we patch the name the
    manager actually calls.
    """
    import src.authentication.manager as manager_mod

    cfg = SimpleNamespace(
        auth=SimpleNamespace(
            allow_training_fallback_policy=allow_fb,
            decision_strategy="k",
            vote_window_size=7,
            vote_min_rejects=6,
            ema_alpha=0.25,
            secondary_hysteresis_enabled=False,
        )
    )
    monkeypatch.setattr(manager_mod, "get_ca_config", lambda *a, **k: cfg)


# ---------------------------------------------------------------------------
# Ready / not-ready gating.
# ---------------------------------------------------------------------------


def test_check_ready_best_policy_ok(tmp_path, monkeypatch) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    _write_policy(models_root, user, name="best_lock_policy.json", extra=_ready_extra())

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    assert ok is True
    assert reason == ""


def test_check_legacy_unmarked_policy_not_ready(tmp_path, monkeypatch) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    # No readiness fields, no grid_search, no policy-search threshold_strategy.
    _write_policy(models_root, user, name="best_lock_policy.json", extra={})

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    assert ok is False
    assert reason.startswith("policy_not_ready")


def test_check_degrade_switch_accepts_training_fallback(tmp_path, monkeypatch) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=True)
    # ONLY training_fallback_policy.json present (no best_lock_policy.json).
    _write_policy(
        models_root,
        user,
        name="training_fallback_policy.json",
        extra={
            "interrupt_rule": "ema",
            "decision_strategy": "ema",
            "policy_status": "training_fallback",
            "policy_search_completed": False,
            "threshold_strategy": "training_val_genuine_threshold",
        },
    )

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    assert ok is True
    assert reason == ""


def test_check_training_fallback_rejected_when_switch_off(tmp_path, monkeypatch) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    _write_policy(
        models_root,
        user,
        name="training_fallback_policy.json",
        extra={
            "interrupt_rule": "ema",
            "decision_strategy": "ema",
            "policy_status": "training_fallback",
            "policy_search_completed": False,
            "threshold_strategy": "training_val_genuine_threshold",
        },
    )

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    # Switch off => best_lock_policy.json is required and missing.
    assert ok is False


# ---------------------------------------------------------------------------
# Threshold validation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_threshold",
    [float("inf"), float("-inf"), float("nan"), sys.float_info.max, 1e7],
)
def test_check_rejects_non_finite_threshold(tmp_path, monkeypatch, bad_threshold) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    extra = _ready_extra()
    extra["threshold"] = bad_threshold
    _write_policy(models_root, user, name="best_lock_policy.json", extra=extra)

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    assert ok is False
    assert reason != ""


# ---------------------------------------------------------------------------
# Required-field validation.
# ---------------------------------------------------------------------------


def test_check_missing_threshold_strategy(tmp_path, monkeypatch) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    extra = _ready_extra()
    extra["threshold_strategy"] = ""  # otherwise ready
    _write_policy(models_root, user, name="best_lock_policy.json", extra=extra)

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    assert ok is False
    assert reason == "missing threshold_strategy"


# ---------------------------------------------------------------------------
# Genuine-score band validation.
# ---------------------------------------------------------------------------


def test_check_genuine_band_rejects_far_threshold(tmp_path, monkeypatch) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    extra = _ready_extra()
    # Genuine band is tight around [-0.20, -0.10]; threshold far below it.
    extra["threshold"] = -50.0
    extra["genuine_score_stats"] = {
        "source": "val",
        "count": 100,
        "min": -0.20,
        "max": -0.10,
        "mean": -0.15,
        "std": 0.02,
    }
    _write_policy(models_root, user, name="best_lock_policy.json", extra=extra)

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    assert ok is False
    assert "genuine" in reason


def test_check_genuine_band_accepts_in_band_threshold(tmp_path, monkeypatch) -> None:
    user = "device-a"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    extra = _ready_extra()
    extra["threshold"] = -0.15  # inside [min, max]
    extra["genuine_score_stats"] = {
        "source": "val",
        "count": 100,
        "min": -0.20,
        "max": -0.10,
        "mean": -0.15,
        "std": 0.02,
    }
    _write_policy(models_root, user, name="best_lock_policy.json", extra=extra)

    manager = AuthSessionManager(models_root=models_root)
    ok, reason = manager.check_trained_model(user)

    assert ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Session-start recording (contract E).
# ---------------------------------------------------------------------------


def test_start_session_writes_model_validation_json(tmp_path, monkeypatch) -> None:
    user = "device-a"
    session_id = "auth-session"
    models_root = _seed_artifacts(tmp_path, monkeypatch, user)
    _patch_ca_config(monkeypatch, allow_fb=False)
    _write_policy(models_root, user, name="best_lock_policy.json", extra=_ready_extra())

    manager = AuthSessionManager(models_root=models_root)
    accepted, message, policy = manager.start_session(user, session_id)

    assert accepted is True, message
    assert policy is not None

    validation_path = (
        Path(settings.inference_storage_path) / user / session_id / "model_validation.json"
    )
    assert validation_path.exists()
    details = json.loads(validation_path.read_text(encoding="utf-8"))
    assert "artifacts" in details
    assert details["policy_status"] == "ready"
    assert details["policy_source"] == "best"
    assert set(details["artifacts"].keys()) == {"checkpoint", "config", "scaler", "policy"}
