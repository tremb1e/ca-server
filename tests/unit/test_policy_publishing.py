"""Task 2 — policy publishing flow: load_best_policy readiness shim, genuine
score stats, and atomic best-policy writes. (No secondary-hysteresis content.)"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.authentication.runner import load_best_policy
from src.policy_search.runner import _atomic_write_json, _genuine_score_stats


def _write_policy(models_root: Path, user: str, extra: dict, *, filename: str = "best_lock_policy.json") -> Path:
    user_dir = models_root / user
    (user_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (user_dir / "checkpoints" / "vqgan.pt").write_bytes(b"x")
    (user_dir / "checkpoints" / "vqgan.json").write_text(
        json.dumps({"input_height": 6, "input_width": 20}), encoding="utf-8"
    )
    payload = {
        "user": user,
        "window": 0.2,
        "overlap": 0.5,
        "target_width": 20,
        "threshold": -0.12,
        "interrupt_rule": "ema",
        "decision_strategy": "ema",
        "ema_alpha": 0.25,
        "vqgan_checkpoint": "checkpoints/vqgan.pt",
        "vqgan_config": "checkpoints/vqgan.json",
        "model_version": "vqgan.pt",
    }
    payload.update(extra)
    path = user_dir / filename
    path.write_text(json.dumps({user: payload}), encoding="utf-8")
    return path


READY = {
    "policy_status": "ready",
    "policy_search_completed": True,
    "score_metric": "mse",
    "score_scale": "negative_reconstruction_error",
    "threshold_strategy": "interrupt_window_frr",
}


def test_load_best_policy_reads_explicit_ready_fields(tmp_path) -> None:
    models_root = tmp_path / "models"
    _write_policy(models_root, "u1", READY)
    cfg = load_best_policy("u1", models_root=models_root)
    assert cfg.policy_status == "ready"
    assert cfg.policy_search_completed is True
    assert cfg.policy_source == "best"
    assert cfg.score_metric == "mse"
    assert cfg.score_scale == "negative_reconstruction_error"
    assert cfg.threshold_strategy == "interrupt_window_frr"
    assert cfg.policy_file.endswith("best_lock_policy.json")


def test_load_best_policy_infers_legacy_grid_search_as_ready(tmp_path) -> None:
    models_root = tmp_path / "models"
    _write_policy(models_root, "u1", {"grid_search": {"grid_results_csv": "x.csv"}})
    cfg = load_best_policy("u1", models_root=models_root)
    assert cfg.policy_status == "ready"
    assert cfg.policy_search_completed is True


def test_load_best_policy_infers_bare_legacy_as_not_ready(tmp_path) -> None:
    models_root = tmp_path / "models"
    _write_policy(models_root, "u1", {})
    cfg = load_best_policy("u1", models_root=models_root)
    assert cfg.policy_status == "legacy"
    assert cfg.policy_search_completed is False


def test_fallback_requires_switch_and_best_wins(tmp_path) -> None:
    models_root = tmp_path / "models"
    _write_policy(
        models_root,
        "u1",
        {"policy_status": "training_fallback", "policy_search_completed": False},
        filename="training_fallback_policy.json",
    )
    # Switch OFF: no authoritative best policy -> error.
    with pytest.raises(FileNotFoundError):
        load_best_policy("u1", models_root=models_root, allow_training_fallback=False)
    # Switch ON: the fallback is loaded.
    cfg = load_best_policy("u1", models_root=models_root, allow_training_fallback=True)
    assert cfg.policy_source == "training_fallback"
    assert cfg.policy_status == "training_fallback"
    assert cfg.policy_search_completed is False

    # Best policy must win over the fallback even with the switch ON.
    _write_policy(models_root, "u1", READY)
    cfg = load_best_policy("u1", models_root=models_root, allow_training_fallback=True)
    assert cfg.policy_source == "best"
    assert cfg.policy_status == "ready"


def test_genuine_score_stats_shape() -> None:
    scores = np.array([0.1, 0.2, 0.3, -1.0, -2.0], dtype=np.float32)
    labels = np.array([1, 1, 1, 0, 0], dtype=np.int8)
    stats = _genuine_score_stats(scores, labels)
    assert stats is not None
    assert stats["source"] == "val"
    assert stats["count"] == 3
    assert stats["min"] == pytest.approx(0.1, abs=1e-5)
    assert stats["max"] == pytest.approx(0.3, abs=1e-5)
    assert set(stats["quantiles"].keys()) == {"0.001", "0.01", "0.05", "0.5", "0.95", "0.99", "0.999"}


def test_genuine_score_stats_none_without_genuine() -> None:
    scores = np.array([0.1, 0.2], dtype=np.float32)
    labels = np.array([0, 0], dtype=np.int8)
    assert _genuine_score_stats(scores, labels) is None


def test_atomic_write_json_leaves_no_tmp(tmp_path) -> None:
    path = tmp_path / "out" / "best.json"
    _atomic_write_json(path, {"a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}
    _atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 2}
    assert list(path.parent.glob("*.tmp")) == []
