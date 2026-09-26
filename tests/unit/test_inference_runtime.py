import asyncio
import csv
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.authentication import runner
from src.authentication.manager import AuthSessionManager, AuthSessionState, VQGANModelCache
from src.authentication.vqgan_inference import score_windows
from src.policy_search import runner as search


def test_scoring_disables_autograd_inside_worker_thread():
    from src.utils.accelerator import resolve_torch_device

    device = resolve_torch_device(os.environ.get("VQGAN_TEST_DEVICE", "cpu"))
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.grad_enabled = []

        def forward(self, batch):
            self.grad_enabled.append(torch.is_grad_enabled())
            return batch * self.weight, None, None

    model = Decoder().to(device)
    windows = np.ones((3, 1, 6, 20), dtype=np.float32)

    async def invoke():
        return await asyncio.to_thread(
            score_windows, model, windows, device=device, use_amp=False, batch_size=2,
        )

    with torch.enable_grad():
        scores = asyncio.run(invoke())
        assert torch.is_grad_enabled()
    np.testing.assert_allclose(scores, [-1, -1, -1])
    assert model.grad_enabled == [False, False]
    assert model.weight.grad is None


@pytest.mark.parametrize("changed_file", ["vqgan_checkpoint", "vqgan_config"])
def test_model_cache_reloads_replaced_artifacts(tmp_path, monkeypatch, changed_file):
    from src.authentication import vqgan_inference
    from src.utils import accelerator

    policy = SimpleNamespace(user="u", vqgan_checkpoint=tmp_path / "m.pt", vqgan_config=tmp_path / "m.json")
    policy.vqgan_checkpoint.write_bytes(b"model")
    policy.vqgan_config.write_text("{}")
    loaded = []

    def load(*args, **kwargs):
        model = object()
        loaded.append(model)
        return model

    monkeypatch.setattr(vqgan_inference, "load_vqgan", load)
    monkeypatch.setattr(accelerator, "resolve_torch_device", lambda _: torch.device("cpu"))
    cache = VQGANModelCache(max_models=1)
    first = cache.get(policy)
    assert cache.get(policy) is first
    artifact = getattr(policy, changed_file)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(artifact.read_bytes())
    replacement.replace(artifact)
    assert cache.get(policy) is not first
    assert len(loaded) == 2
    assert cache.snapshot()["loaded_count"] == 1


def test_offline_replay_honors_metric_and_exact_window_limit(tmp_path, monkeypatch):
    from src.authentication import vqgan_inference
    from src.utils.ca_train import ensure_ca_train_on_path

    ensure_ca_train_on_path()
    import hmog_data

    csv_path = tmp_path / "input.csv"
    csv_path.write_text("placeholder")
    policy = runner.AuthRunConfig(
        user="u", window_size=0.2, target_width=20, overlap=0.5, threshold=-1.0,
        interrupt_rule="none", decision_strategy="none", k_rejects=0, vote_window_size=0,
        vote_min_rejects=0, ema_alpha=0.25, vqgan_checkpoint=tmp_path / "m.pt",
        vqgan_config=tmp_path / "m.json", score_metric="l1",
    )
    monkeypatch.setattr(vqgan_inference, "load_vqgan", lambda *a, **kw: object())
    monkeypatch.setattr(
        hmog_data, "iter_windows_from_csv_unlabeled_with_session",
        lambda *a, **kw: ((i, "u", "s", np.ones((1, 6, 20))) for i in range(10)),
    )
    metrics = []

    def score(model, windows, **kwargs):
        metrics.append(kwargs["score_metric"])
        return np.zeros(len(windows))

    monkeypatch.setattr(vqgan_inference, "score_windows", score)
    output, _ = runner.run_auth_inference(
        csv_path=csv_path, policy=policy, device="cpu", output_csv=tmp_path / "out.csv", max_windows=1,
    )
    with output.open() as handle:
        assert len(list(csv.DictReader(handle))) == 1
    assert metrics == ["l1"]
    with pytest.raises(ValueError, match="max_windows"):
        runner.run_auth_inference(csv_path=csv_path, policy=policy, max_windows=0)


@pytest.mark.parametrize("changed", [{"device": "npu:0"}, {"use_amp": True}])
def test_policy_cache_recomputes_after_execution_precision_change(tmp_path, monkeypatch, changed):
    artifacts = [tmp_path / "input.csv", tmp_path / "m.pt", tmp_path / "m.json"]
    for path in artifacts:
        path.write_text("{}")
    calls = []

    def score(**kwargs):
        calls.append(kwargs)
        return search.ScoreArrays(np.array([0]), np.array([1]), np.array([-1.0]))

    monkeypatch.setattr(search, "_score_split_inprocess", score)
    kwargs = dict(
        user="u", split="val", auth_method="vqgan-only", window_size=0.2, overlap=0.5,
        target_width=20, csv_path=artifacts[0], vqgan_ckpt=artifacts[1], lm_ckpt=tmp_path / "lm.pt",
        device="cpu", token_batch_size=8, use_amp=False, cache_dir=tmp_path / "cache",
    )
    search._load_or_score_split(**kwargs)
    search._load_or_score_split(**kwargs)
    assert len(calls) == 1
    search._load_or_score_split(**(kwargs | changed))
    assert len(calls) == 2


def test_online_records_ignore_retries_and_reset_only_with_new_boot_evidence():
    from src.utils.reject_trackers import EMAScoreTracker

    state = AuthSessionState(user_id="u", session_id="s", policy=SimpleNamespace())
    state.ema_rejects = EMAScoreTracker()
    state.ema_rejects.update(-1, alpha=0.5, threshold=0, reset_on_interrupt=False)
    state.emit_gate.pending = 4

    def records(*timestamps):
        return {"acc": [{"timestamp": ts, "x": 1} for ts in timestamps]}

    first = AuthSessionManager._fresh_sensor_records(
        state, records(1000, 1010, 1010), {"base_wall_ms": 10000, "device_uptime_ns": 1010000000},
    )
    assert [r["timestamp"] for r in first["acc"]] == [1000, 1010]
    state.tail_records = first
    stale = AuthSessionManager._fresh_sensor_records(
        state, records(990, 1000), {"base_wall_ms": 9990, "device_uptime_ns": 1000000000},
    )
    assert not any(stale.values())
    assert state.tail_records is first
    overlap = AuthSessionManager._fresh_sensor_records(state, records(1000, 1010, 1020), {})
    assert [r["timestamp"] for r in overlap["acc"]] == [1020]
    reboot = AuthSessionManager._fresh_sensor_records(
        state, records(10, 20), {"base_wall_ms": 11000, "device_uptime_ns": 20000000},
    )
    assert [r["timestamp"] for r in reboot["acc"]] == [10, 20]
    assert not any(state.tail_records.values())
    assert state.ema_rejects.ema_score is None
    assert state.emit_gate.pending == 0
