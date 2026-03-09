import json
from pathlib import Path

from src.authentication.runner import load_best_policy


def test_load_best_policy_resolves_relative_paths(tmp_path) -> None:
    user = "user_a"
    models_root = tmp_path / "data_storage" / "models"
    user_dir = models_root / user
    ckpt = user_dir / "checkpoints" / "model.pt"
    cfg = user_dir / "checkpoints" / "model.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("ok", encoding="utf-8")
    cfg.write_text("{}", encoding="utf-8")

    policy_path = user_dir / "best_lock_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                user: {
                    "user": user,
                    "window": 0.2,
                    "overlap": 0.5,
                    "target_width": 20,
                    "threshold": -0.1,
                    "k_rejects": 3,
                    "vqgan_checkpoint": "checkpoints/model.pt",
                    "vqgan_config": "checkpoints/model.json",
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_best_policy(user, models_root=models_root)

    assert loaded.vqgan_checkpoint == ckpt.resolve()
    assert loaded.vqgan_config == cfg.resolve()


def test_load_best_policy_rebases_legacy_absolute_paths(tmp_path) -> None:
    user = "user_a"
    server_root = tmp_path
    models_root = server_root / "data_storage" / "models"
    user_dir = models_root / user
    ckpt = user_dir / "checkpoints" / "model.pt"
    cfg = user_dir / "checkpoints" / "model.json"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_text("ok", encoding="utf-8")
    cfg.write_text("{}", encoding="utf-8")

    legacy_root = Path("/old/worktree/ca-server")
    legacy_ckpt = legacy_root / "data_storage" / "models" / user / "checkpoints" / "model.pt"
    legacy_cfg = legacy_root / "data_storage" / "models" / user / "checkpoints" / "model.json"

    policy_path = user_dir / "best_lock_policy.json"
    policy_path.write_text(
        json.dumps(
            {
                user: {
                    "user": user,
                    "window": 0.2,
                    "overlap": 0.5,
                    "target_width": 20,
                    "threshold": -0.1,
                    "k_rejects": 3,
                    "vqgan_checkpoint": str(legacy_ckpt),
                    "vqgan_config": str(legacy_cfg),
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_best_policy(user, models_root=models_root, policy_path=policy_path)

    assert loaded.vqgan_checkpoint == ckpt.resolve()
    assert loaded.vqgan_config == cfg.resolve()
