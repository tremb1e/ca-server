import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.training import runner as training_runner
from src.training.runner import TrainingCommandError, _read_best_window, _run_training_command, run_window_sweep_for_user


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
        auth=SimpleNamespace(max_decision_time_sec=2.0),
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        del kwargs
        calls.append([str(x) for x in cmd])
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_log_dir = Path(cmd[cmd.index("--log-dir") + 1])
        users_arg = next(part for part in cmd if str(part).startswith("--users="))
        run_user = str(users_arg).split("=", 1)[1]
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

        return SimpleNamespace(returncode=0)

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
    assert f"--users={user_id}" in calls[0]
    assert len(results) == 1
    assert results[0].summary["user"] == user_id

    # Training writes the training_fallback_policy.json (NOT best_lock_policy.json);
    # the formal best_lock_policy.json is only produced by policy search.
    policy = json.loads(
        (models_root / user_id / "training_fallback_policy.json").read_text(encoding="utf-8")
    )
    assert user_id in policy
    assert policy[user_id]["vqgan_checkpoint"] == f"checkpoints/vqgan_user_{user_id}_ws_{ws:.1f}.pt"
    assert policy[user_id]["vqgan_config"] == f"checkpoints/vqgan_user_{user_id}_ws_{ws:.1f}.json"
    # New training-fallback readiness/telemetry fields (contract A.1).
    assert policy[user_id]["policy_status"] == "training_fallback"
    assert policy[user_id]["policy_search_completed"] is False
    assert policy[user_id]["score_metric"] == "mse"
    assert policy[user_id]["score_scale"] == "negative_reconstruction_error"

    summary_rows = json.loads((models_root / user_id / "training_summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary_rows, list)
    assert summary_rows[0]["user"] == user_id


def test_run_window_sweep_passes_dash_prefixed_user_with_equals(tmp_path, monkeypatch) -> None:
    user_id = "-OzOz6DF-4eSwCnStI2kQBpbQTkWN6ODZSxyQv5MAAI="
    ws = 0.2
    dataset_path = tmp_path / "dataset"
    models_root = tmp_path / "models"
    script_path = tmp_path / "fake_train.py"
    dataset_path.mkdir(parents=True)
    script_path.write_text("# fake", encoding="utf-8")

    fake_ca_cfg = SimpleNamespace(
        windows=SimpleNamespace(sizes=[ws], sampling_rate_hz=100, overlap=0.5),
        auth=SimpleNamespace(max_decision_time_sec=2.0),
        training=SimpleNamespace(
            max_epochs=1,
            batch_size=4,
            early_stop_patience=0,
            max_parallel_train=1,
            run_policy_search=False,
        ),
    )
    calls: list[list[str]] = []

    def _fake_run(cmd, **kwargs):
        del kwargs
        calls.append([str(x) for x in cmd])
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_log_dir = Path(cmd[cmd.index("--log-dir") + 1])
        users_arg = next(part for part in cmd if str(part).startswith("--users="))
        run_user = str(users_arg).split("=", 1)[1]
        run_ws = float(cmd[cmd.index("--window-sizes") + 1])
        ckpt = out_dir / "checkpoints" / f"vqgan_user_{run_user}_ws_{run_ws:.1f}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text("ok", encoding="utf-8")
        run_log_dir.mkdir(parents=True, exist_ok=True)
        (run_log_dir / "best_windows.json").write_text(
            json.dumps(
                {
                    run_user: {
                        "user": run_user,
                        "window": run_ws,
                        "val": {"threshold": -0.12},
                        "checkpoint": str(ckpt),
                    }
                }
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(training_runner.subprocess, "run", _fake_run)
    monkeypatch.setattr(training_runner, "k_from_interrupt_time", lambda *args, **kwargs: 3)

    results = run_window_sweep_for_user(
        user_id,
        device="cpu",
        ca_cfg=fake_ca_cfg,
        dataset_path=dataset_path,
        models_root=models_root,
        ca_train_script=script_path,
    )

    assert len(results) == 1
    assert f"--users={user_id}" in calls[0]
    assert "--users" not in calls[0]
    # Training writes training_fallback_policy.json (contract A.1).
    policy = json.loads(
        (models_root / user_id / "training_fallback_policy.json").read_text(encoding="utf-8")
    )
    assert user_id in policy
    assert policy[user_id]["policy_status"] == "training_fallback"
    assert policy[user_id]["policy_search_completed"] is False
    assert policy[user_id]["score_metric"] == "mse"
    assert policy[user_id]["score_scale"] == "negative_reconstruction_error"


def test_training_command_error_includes_subprocess_log_context(tmp_path, monkeypatch) -> None:
    log_dir = tmp_path / "logs"

    def _fake_run(cmd, stdout, stderr, check):
        del cmd, stderr, check
        stdout.write(b"usage: hmog_vqgan_experiment.py [-h]\\n")
        stdout.write(b"hmog_vqgan_experiment.py: error: unrecognized arguments: -Oz...\\n")
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(training_runner.subprocess, "run", _fake_run)

    with pytest.raises(TrainingCommandError) as exc_info:
        _run_training_command(["python", "hmog_vqgan_experiment.py"], log_dir=log_dir)

    message = str(exc_info.value)
    assert "exit code 2" in message
    assert "unrecognized arguments" in message
    assert (log_dir / "hmog_vqgan_subprocess.log").exists()
