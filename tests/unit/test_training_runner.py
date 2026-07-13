import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.training import runner as training_runner
from src.training.runner import _explicit_accelerator_index, _resolve_user_input_height, _vqgan_config
from src.training.runner import _read_best_window, run_window_sweep_for_user


def test_read_best_window_returns_user_record(tmp_path) -> None:
    log_dir = tmp_path
    payload = {"user_a": {"window": 0.2, "checkpoint": "/tmp/model.pt"}}
    (log_dir / "best_windows.json").write_text(json.dumps(payload), encoding="utf-8")

    record = _read_best_window(log_dir, "user_a")
    assert record["window"] == 0.2


def test_read_best_window_reports_training_log_context(tmp_path) -> None:
    log_dir = tmp_path
    (log_dir / "best_windows.json").write_text("{}", encoding="utf-8")
    (log_dir / "hmog_vqgan.log").write_text(
        "2026-02-26 20:07:10,934 [ERROR] val split must contain both classes\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError) as exc_info:
        _read_best_window(log_dir, "user_a")

    message = str(exc_info.value)
    assert "summary is empty" in message
    assert "must contain both classes" in message


def test_run_window_sweep_retrains_when_cached_summary_has_other_user(tmp_path, monkeypatch) -> None:
    user_id = "user_target"
    other_user = "user_other"
    ws = 0.2

    dataset_path = tmp_path / "dataset"
    models_root = tmp_path / "models"
    log_dir = models_root / user_id / "logs" / "ws_0.2"
    script_path = tmp_path / "fake_train.py"

    dataset_path.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    script_path.write_text("# fake", encoding="utf-8")

    (log_dir / "best_windows.json").write_text(
        json.dumps({other_user: {"window": ws, "checkpoint": "/tmp/other.pt"}}),
        encoding="utf-8",
    )

    fake_ca_cfg = SimpleNamespace(
        windows=SimpleNamespace(sizes=[ws], sampling_rate_hz=100, overlap=0.5),
        auth=SimpleNamespace(max_decision_time_sec=2.0, ema_alpha=0.25, allow_training_fallback_policy=False),
        training=SimpleNamespace(run_policy_search=False),
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append([str(x) for x in cmd])
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_log_dir = Path(cmd[cmd.index("--log-dir") + 1])
        run_user = next(a.split("=", 1)[1] for a in cmd if a.startswith("--users="))
        run_ws = float(cmd[cmd.index("--window-sizes") + 1])

        ckpt = out_dir / "checkpoints" / f"vqgan_user_{run_user}_ws_{run_ws:.1f}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("ok", encoding="utf-8")

        payload = {
            run_user: {
                "user": run_user,
                "window": run_ws,
                "val": {"threshold": -0.12},
                "checkpoint": str(ckpt),
            }
        }
        run_log_dir.mkdir(parents=True, exist_ok=True)
        (run_log_dir / "best_windows.json").write_text(json.dumps(payload), encoding="utf-8")

        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(training_runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(training_runner, "k_from_interrupt_time", lambda *args, **kwargs: 3)

    results = run_window_sweep_for_user(
        user_id,
        device="cpu",
        window_sizes=[ws],
        vqgan_epochs=1,
        reuse_checkpoints=True,
        ca_cfg=fake_ca_cfg,
        dataset_path=dataset_path,
        models_root=models_root,
        ca_train_script=script_path,
    )

    assert len(calls) == 1
    # 横杠开头的设备 ID 必须以 --users=<id> 形式传入，避免被 argparse 当作选项。
    assert f"--users={user_id}" in calls[0]
    assert "--users" not in calls[0]
    assert len(results) == 1
    assert results[0].summary["user"] == user_id

    # 训练阶段只写训练兜底策略，不直接产出正式 best_lock_policy.json。
    fallback_path = models_root / user_id / "training_fallback_policy.json"
    assert fallback_path.exists()
    assert not (models_root / user_id / "best_lock_policy.json").exists()
    policy = json.loads(fallback_path.read_text(encoding="utf-8"))
    assert user_id in policy
    assert policy[user_id]["policy_status"] == "training_fallback"
    assert policy[user_id]["policy_search_completed"] is False
    assert policy[user_id]["score_metric"] == "mse"
    assert policy[user_id]["score_scale"] == "negative_reconstruction_error"
    assert policy[user_id]["vqgan_checkpoint"] == f"checkpoints/vqgan_user_{user_id}_ws_{ws:.1f}.pt"
    assert policy[user_id]["vqgan_config"] == f"checkpoints/vqgan_user_{user_id}_ws_{ws:.1f}.json"

    summary_rows = json.loads((models_root / user_id / "training_summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary_rows, list)
    assert summary_rows[0]["user"] == user_id


def test_run_window_sweep_does_not_reuse_stale_channel_checkpoint(tmp_path, monkeypatch) -> None:
    # A leftover 9-axis checkpoint must NOT be reused for the current 6-axis model,
    # even with reuse_checkpoints=True (the auto-retrain default). It must retrain.
    user_id = "user_stale"
    ws = 0.2
    dataset_path = tmp_path / "dataset"
    models_root = tmp_path / "models"
    log_dir = models_root / user_id / "logs" / "ws_0.2"
    script_path = tmp_path / "fake_train.py"
    dataset_path.mkdir(parents=True)
    log_dir.mkdir(parents=True)
    script_path.write_text("# fake", encoding="utf-8")

    stale_ckpt = models_root / user_id / "checkpoints" / f"vqgan_user_{user_id}_ws_0.2.pt"
    stale_ckpt.parent.mkdir(parents=True, exist_ok=True)
    stale_ckpt.write_text("stale", encoding="utf-8")
    # Sidecar config marks the cached checkpoint as legacy 9-axis.
    stale_ckpt.with_suffix(".json").write_text(
        json.dumps({"input_height": 9, "input_width": 20}), encoding="utf-8"
    )
    (log_dir / "best_windows.json").write_text(
        json.dumps({user_id: {"window": ws, "checkpoint": str(stale_ckpt)}}), encoding="utf-8"
    )

    fake_ca_cfg = SimpleNamespace(
        windows=SimpleNamespace(sizes=[ws], sampling_rate_hz=100, overlap=0.5),
        auth=SimpleNamespace(max_decision_time_sec=2.0, ema_alpha=0.25, allow_training_fallback_policy=False),
        training=SimpleNamespace(run_policy_search=False),
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        calls.append([str(x) for x in cmd])
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_log_dir = Path(cmd[cmd.index("--log-dir") + 1])
        run_user = next(a.split("=", 1)[1] for a in cmd if a.startswith("--users="))
        run_ws = float(cmd[cmd.index("--window-sizes") + 1])
        ckpt = out_dir / "checkpoints" / f"vqgan_user_{run_user}_ws_{run_ws:.1f}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("fresh", encoding="utf-8")
        run_log_dir.mkdir(parents=True, exist_ok=True)
        (run_log_dir / "best_windows.json").write_text(
            json.dumps({run_user: {"user": run_user, "window": run_ws, "val": {"threshold": -0.1}, "checkpoint": str(ckpt)}}),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(training_runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(training_runner, "k_from_interrupt_time", lambda *a, **k: 3)

    results = run_window_sweep_for_user(
        user_id,
        device="cpu",
        window_sizes=[ws],
        vqgan_epochs=1,
        reuse_checkpoints=True,
        ca_cfg=fake_ca_cfg,
        dataset_path=dataset_path,
        models_root=models_root,
        ca_train_script=script_path,
    )

    assert len(calls) == 1  # stale 9-axis checkpoint forced a fresh retrain
    assert len(results) == 1
    cfg = json.loads(
        (models_root / user_id / "checkpoints" / f"vqgan_user_{user_id}_ws_0.2.json").read_text(encoding="utf-8")
    )
    assert cfg["input_height"] == 6


def test_sensor_mode_controls_vqgan_input_height(tmp_path) -> None:
    dataset = tmp_path / "processed" / "window"
    mode_dir = tmp_path / "processed" / "z-score" / "u9"
    dataset.mkdir(parents=True)
    mode_dir.mkdir(parents=True)
    (mode_dir / "sensor_mode.json").write_text(
        json.dumps({"input_height": 9}), encoding="utf-8"
    )
    assert _resolve_user_input_height("u9", dataset) == 9
    cfg = _vqgan_config(
        20,
        input_height=9,
        base_channels=96,
        latent_dim=256,
        codebook_vectors=512,
        beta=0.25,
    )
    assert cfg["input_height"] == 9


@pytest.mark.parametrize(
    ("device", "expected"),
    [("npu:0", 0), ("npu:7", 7), ("cuda:2", 2), ("cpu", None), ("auto", None)],
)
def test_explicit_accelerator_index_is_forwarded_to_worker_pool(device, expected) -> None:
    assert _explicit_accelerator_index(device) == expected
