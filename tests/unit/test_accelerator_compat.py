from types import SimpleNamespace

import pytest
import torch

from ca_train import accelerator as training
from src.utils import accelerator as runtime


@pytest.mark.parametrize("adapter", [training, runtime])
def test_npu_without_native_autocast_runs_full_precision(monkeypatch, adapter):
    monkeypatch.setattr(adapter, "try_import_torch_npu", lambda: False)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(), raising=False)

    def unsupported_autocast(*args, **kwargs):
        raise AssertionError("Generic autocast must not handle this unavailable backend")

    monkeypatch.setattr(adapter.amp, "autocast", unsupported_autocast)
    value = torch.tensor([2.0], requires_grad=True)
    with adapter.autocast_context(SimpleNamespace(type="npu"), enabled=True):
        loss = value.square().sum()
    loss.backward()
    assert value.grad.item() == 4.0
    assert loss.dtype == torch.float32


def test_npu_without_native_scaler_can_complete_optimizer_step(monkeypatch):
    monkeypatch.setattr(training, "try_import_torch_npu", lambda: False)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(), raising=False)
    scaler = training.make_grad_scaler(SimpleNamespace(type="npu"), enabled=True)
    parameter = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaler.scale(parameter.square().sum()).backward()
    scaler.step(optimizer)
    scaler.update()
    assert not scaler.is_enabled()
    assert parameter.item() == pytest.approx(1.6)
