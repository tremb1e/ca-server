import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from src.training import runner as training_runner
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
        auth=SimpleNamespace(max_decision_time_sec=2.0),
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, check):
        del check
        calls.append([str(x) for x in cmd])
        out_dir = Path(cmd[cmd.index("--output-dir") + 1])
        run_log_dir = Path(cmd[cmd.index("--log-dir") + 1])
        run_user = str(cmd[cmd.index("--users") + 1])
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

    monkeypatch.setattr(training_runner, "ensure_ca_train_on_path", lambda: None)
    monkeypatch.setattr(training_runner.subprocess, "run", _fake_run)
    monkeypatch.setitem(
        sys.modules,
        "hmog_consecutive_rejects",
        SimpleNamespace(k_from_interrupt_time=lambda *args, **kwargs: 3),
    )

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
    assert len(results) == 1
    assert results[0].summary["user"] == user_id

    policy = json.loads((models_root / user_id / "best_lock_policy.json").read_text(encoding="utf-8"))
    assert user_id in policy

    summary_rows = json.loads((models_root / user_id / "training_summary.json").read_text(encoding="utf-8"))
    assert isinstance(summary_rows, list)
    assert summary_rows[0]["user"] == user_id
