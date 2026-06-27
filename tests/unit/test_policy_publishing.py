"""Functional coverage for the policy-publishing refactor.

Covers (see .ca_refactor_contract.md sections A/C):
  1. load_best_policy readiness shim + file selection.
  5. policy_search helpers (_genuine_score_stats, _atomic_write_json).
  6. training writes training_fallback_policy.json with the new fields.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from src.authentication.runner import load_best_policy


def _write_policy(path: Path, user: str, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "user": user,
        "window": 0.2,
        "overlap": 0.5,
        "target_width": 20,
        "threshold": -0.1,
        "k_rejects": 3,
        "vqgan_checkpoint": "checkpoints/model.pt",
        "vqgan_config": "checkpoints/model.json",
    }
    base.update(extra)
    path.write_text(json.dumps({user: base}), encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. load_best_policy readiness shim (contract C).
# ---------------------------------------------------------------------------


def test_load_best_policy_honors_explicit_status(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "models"
    policy_path = models_root / user / "best_lock_policy.json"
    _write_policy(
        policy_path,
        user,
        {
            "policy_status": "ready",
            "policy_search_completed": True,
            "threshold_strategy": "interrupt_window_frr",
            "score_metric": "mse",
            "score_scale": "negative_reconstruction_error",
        },
    )

    cfg = load_best_policy(user, models_root=models_root)

    assert cfg.policy_status == "ready"
    assert cfg.policy_search_completed is True
    assert cfg.policy_source == "best"
    assert cfg.threshold_strategy == "interrupt_window_frr"
    assert cfg.score_metric == "mse"
    assert cfg.score_scale == "negative_reconstruction_error"
    assert cfg.policy_file == str(policy_path)


def test_load_best_policy_explicit_completed_only_infers_status(tmp_path) -> None:
    # When only policy_search_completed is present, status is inferred from it.
    user = "user_a"
    models_root = tmp_path / "models"
    policy_path = models_root / user / "best_lock_policy.json"
    _write_policy(policy_path, user, {"policy_search_completed": False})

    cfg = load_best_policy(user, models_root=models_root)

    assert cfg.policy_search_completed is False
    # completed=False and no explicit status => "unknown".
    assert cfg.policy_status == "unknown"


def test_load_best_policy_legacy_grid_search_treated_as_ready(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "models"
    policy_path = models_root / user / "best_lock_policy.json"
    # Legacy genuine policy-search output: has grid_search but no explicit readiness.
    _write_policy(
        policy_path,
        user,
        {"grid_search": {"grid_results_csv": "policy_search/grid.csv"}},
    )

    cfg = load_best_policy(user, models_root=models_root)

    assert cfg.policy_status == "ready"
    assert cfg.policy_search_completed is True
    assert cfg.policy_source == "best"


def test_load_best_policy_legacy_threshold_strategy_treated_as_ready(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "models"
    policy_path = models_root / user / "best_lock_policy.json"
    _write_policy(policy_path, user, {"threshold_strategy": "interrupt_window_frr"})

    cfg = load_best_policy(user, models_root=models_root)

    assert cfg.policy_status == "ready"
    assert cfg.policy_search_completed is True


def test_load_best_policy_legacy_without_markers_is_not_ready(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "models"
    policy_path = models_root / user / "best_lock_policy.json"
    # No readiness fields, no grid_search, no policy-search threshold strategy.
    _write_policy(policy_path, user, {})

    cfg = load_best_policy(user, models_root=models_root)

    assert cfg.policy_status == "legacy"
    assert cfg.policy_search_completed is False
    assert cfg.policy_source == "best"


def test_load_best_policy_training_fallback_status(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "models"
    fallback_path = models_root / user / "training_fallback_policy.json"
    _write_policy(
        fallback_path,
        user,
        {
            "policy_status": "training_fallback",
            "policy_search_completed": False,
            "threshold_strategy": "training_val_genuine_threshold",
        },
    )

    cfg = load_best_policy(
        user, models_root=models_root, allow_training_fallback=True
    )

    assert cfg.policy_status == "training_fallback"
    assert cfg.policy_search_completed is False
    assert cfg.policy_source == "training_fallback"
    assert cfg.policy_file == str(fallback_path)


def test_load_best_policy_fallback_filename_infers_status_without_fields(tmp_path) -> None:
    # Even without explicit readiness fields, the filename drives the inference.
    user = "user_a"
    models_root = tmp_path / "models"
    fallback_path = models_root / user / "training_fallback_policy.json"
    _write_policy(fallback_path, user, {})

    cfg = load_best_policy(
        user, models_root=models_root, allow_training_fallback=True
    )

    assert cfg.policy_status == "training_fallback"
    assert cfg.policy_search_completed is False
    assert cfg.policy_source == "training_fallback"


def test_load_best_policy_file_selection_requires_allow_flag(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "models"
    # Only the training fallback exists; best_lock_policy.json is absent.
    _write_policy(models_root / user / "training_fallback_policy.json", user, {})

    # Default (allow_training_fallback=False) must look for the (missing) best
    # policy and raise FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        load_best_policy(user, models_root=models_root)

    # With the flag on, the fallback is loaded.
    cfg = load_best_policy(
        user, models_root=models_root, allow_training_fallback=True
    )
    assert cfg.policy_source == "training_fallback"


def test_load_best_policy_prefers_best_over_fallback(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "models"
    _write_policy(
        models_root / user / "best_lock_policy.json",
        user,
        {"policy_status": "ready", "policy_search_completed": True},
    )
    _write_policy(models_root / user / "training_fallback_policy.json", user, {})

    # Even with the flag on, the best policy wins when present.
    cfg = load_best_policy(
        user, models_root=models_root, allow_training_fallback=True
    )
    assert cfg.policy_source == "best"
    assert cfg.policy_status == "ready"


# ---------------------------------------------------------------------------
# 5. policy_search helpers (contract A).
# ---------------------------------------------------------------------------


def test_genuine_score_stats_shape() -> None:
    from src.policy_search.runner import _genuine_score_stats

    scores = np.array([0.1, 0.2, 0.3, -0.5, -0.6], dtype=np.float32)
    labels = np.array([1, 1, 1, 0, 0], dtype=np.int8)

    stats = _genuine_score_stats(scores, labels)

    assert stats is not None
    assert stats["source"] == "val"
    assert stats["count"] == 3
    assert stats["min"] == pytest.approx(0.1, abs=1e-6)
    assert stats["max"] == pytest.approx(0.3, abs=1e-6)
    assert stats["mean"] == pytest.approx(0.2, abs=1e-6)
    assert "std" in stats
    quantiles = stats["quantiles"]
    assert set(quantiles.keys()) == {"0.001", "0.01", "0.05", "0.5", "0.95", "0.99", "0.999"}
    assert all(isinstance(v, float) for v in quantiles.values())
    assert quantiles["0.5"] == pytest.approx(0.2, abs=1e-6)


def test_genuine_score_stats_returns_none_without_genuine() -> None:
    from src.policy_search.runner import _genuine_score_stats

    scores = np.array([-0.5, -0.6, -0.7], dtype=np.float32)
    labels = np.array([0, 0, 0], dtype=np.int8)

    assert _genuine_score_stats(scores, labels) is None


def test_atomic_write_json_writes_valid_json_no_tmp(tmp_path) -> None:
    from src.policy_search.runner import _atomic_write_json

    target = tmp_path / "nested" / "best_lock_policy.json"
    payload = {"user_x": {"threshold": -0.12, "policy_status": "ready"}}

    _atomic_write_json(target, payload)

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    # No leftover .tmp sibling.
    tmp_sibling = target.with_suffix(target.suffix + ".tmp")
    assert not tmp_sibling.exists()
    assert list(target.parent.glob("*.tmp")) == []


def test_atomic_write_json_replaces_existing(tmp_path) -> None:
    from src.policy_search.runner import _atomic_write_json

    target = tmp_path / "best_lock_policy.json"
    _atomic_write_json(target, {"v": 1})
    _atomic_write_json(target, {"v": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 2}
    assert not target.with_suffix(target.suffix + ".tmp").exists()


# ---------------------------------------------------------------------------
# 6. training writes training_fallback_policy.json (explicit unit-level check).
# ---------------------------------------------------------------------------


def test_training_writes_training_fallback_policy(tmp_path, monkeypatch) -> None:
    from types import SimpleNamespace

    import src.training.runner as training_runner

    # _write_vqgan_policy reads get_ca_config().auth.ema_alpha.
    monkeypatch.setattr(
        training_runner,
        "get_ca_config",
        lambda *a, **k: SimpleNamespace(auth=SimpleNamespace(ema_alpha=0.25)),
    )

    user_dir = tmp_path / "models" / "user_a"
    user_dir.mkdir(parents=True, exist_ok=True)
    ckpt = user_dir / "checkpoints" / "vqgan.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("ok", encoding="utf-8")
    cfg = ckpt.with_suffix(".json")
    cfg.write_text("{}", encoding="utf-8")

    policy_path = training_runner._write_vqgan_policy(
        user_dir,
        user_id="user_a",
        window_size=0.2,
        overlap=0.5,
        target_width=20,
        threshold=-0.12,
        k_rejects=3,
        vqgan_checkpoint=ckpt,
        vqgan_config=cfg,
    )

    assert policy_path.name == "training_fallback_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))["user_a"]
    assert policy["policy_status"] == "training_fallback"
    assert policy["policy_search_completed"] is False
    assert policy["score_metric"] == "mse"
    assert policy["score_scale"] == "negative_reconstruction_error"
    assert policy["threshold_strategy"] == "training_val_genuine_threshold"
    assert policy["interrupt_rule"] == "ema"
    assert policy["decision_strategy"] == "ema"
    assert policy["auth_method"] == "vqgan-only"
    assert policy["threshold"] == pytest.approx(-0.12)
