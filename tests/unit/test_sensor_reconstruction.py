import os
import sys

import numpy as np
import pytest
import torch

from ca_train.reconstruction import (
    DEFAULT_SENSOR_WEIGHTS,
    mask_inactive_sensor_inputs,
    reconstruction_errors,
    validate_sensor_weights,
)
from src.authentication.vqgan_inference import score_windows
from src.utils.ca_train import ensure_ca_train_on_path

ensure_ca_train_on_path()
from ca_train import hmog_vqgan_experiment as experiment  # noqa: E402


@pytest.fixture
def device():
    name = os.environ.get("VQGAN_TEST_DEVICE", "cpu")
    return torch.device(name)


class ZeroDecoder(torch.nn.Module):
    def __init__(self, weights=DEFAULT_SENSOR_WEIGHTS):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))
        self.sensor_weights = weights

    def forward(self, batch):
        return self.bias.expand_as(batch), None, self.bias * 0


def sensor_windows(height=9):
    values = np.repeat(np.array([1.0, 2.0, 10.0], dtype=np.float32), 3)[:height]
    first = np.broadcast_to(values[None, :, None], (1, height, 20)).copy()
    return np.stack([first, first * 2])


@pytest.mark.parametrize(("metric", "expected"), [("mse", 2.5), ("l1", 1.5)])
def test_training_and_online_scoring_use_configured_modality_weights(device, metric, expected):
    model = ZeroDecoder().to(device)
    windows = sensor_windows()
    scores = score_windows(model, windows, device=device, use_amp=False, score_metric=metric, batch_size=1)
    factor = 4 if metric == "mse" else 2
    np.testing.assert_allclose(scores, [-expected, -expected * factor], rtol=1e-6)
    loss, rec, _ = experiment.reconstruction_step(
        model, torch.from_numpy(windows), device, False, rec_loss_metric=metric,
    )
    assert rec.item() == pytest.approx(expected * (1 + factor) / 2)
    loss.backward()
    assert torch.isfinite(model.bias.grad)


def test_six_axis_weights_are_renormalized(device):
    scores = score_windows(ZeroDecoder().to(device), sensor_windows(6), device=device, use_amp=False)
    np.testing.assert_allclose(scores, [-2.5, -10.0])


def test_legacy_checkpoint_keeps_original_unweighted_score_scale(device):
    scores = score_windows(ZeroDecoder(None).to(device), sensor_windows(), device=device, use_amp=False)
    np.testing.assert_allclose(scores, [-35.0, -140.0])


def test_weighted_reconstruction_gradient_excludes_magnetometer(device):
    batch = torch.ones((1, 1, 9, 20), device=device)
    decoded = torch.zeros_like(batch, requires_grad=True)
    reconstruction_errors(ZeroDecoder(), batch, decoded).mean().backward()
    group_gradients = decoded.grad.reshape(3, 60).sum(dim=1).cpu()
    torch.testing.assert_close(group_gradients, torch.tensor([-1.0, -1.0, 0.0]))


def test_zero_weight_sensor_is_masked_without_changing_source_window(device):
    batch = torch.from_numpy(sensor_windows()).to(device)
    original = batch.clone()
    masked = mask_inactive_sensor_inputs(batch, DEFAULT_SENSOR_WEIGHTS)
    torch.testing.assert_close(masked[:, :, :6, :], original[:, :, :6, :])
    assert torch.count_nonzero(masked[:, :, 6:, :]) == 0
    torch.testing.assert_close(batch, original)


def test_vqgan_forward_and_encode_mask_zero_weight_sensor_before_encoder(device):
    from ca_train.vqgan import VQGAN

    class SpyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, batch):
            self.seen = batch.detach().clone()
            return batch

    class PassCodebook(torch.nn.Module):
        def forward(self, batch):
            return batch, None, batch.new_zeros(())

    model = VQGAN.__new__(VQGAN)
    torch.nn.Module.__init__(model)
    model.sensor_weights = DEFAULT_SENSOR_WEIGHTS
    model.encoder = SpyEncoder()
    model.quant_conv = torch.nn.Identity()
    model.codebook = PassCodebook()
    model.post_quant_conv = torch.nn.Identity()
    model.decoder = torch.nn.Identity()
    batch = torch.from_numpy(sensor_windows()).to(device)
    for method in (model.forward, model.encode):
        method(batch)
        torch.testing.assert_close(model.encoder.seen[:, :, :6, :], batch[:, :, :6, :])
        assert torch.count_nonzero(model.encoder.seen[:, :, 6:, :]) == 0


@pytest.mark.parametrize("weights", [(0.5, 0.5), (0.4, 0.4, 0.1), (-0.1, 0.6, 0.5), (float("nan"), 0.5, 0.5)])
def test_invalid_sensor_weights_are_rejected(weights):
    with pytest.raises(ValueError, match="sensor_weights"):
        validate_sensor_weights(weights)


@pytest.mark.parametrize(
    ("cli_weights", "expected"),
    [([], DEFAULT_SENSOR_WEIGHTS), (["--sensor-weights", "0.6", "0.3", "0.1"], (0.6, 0.3, 0.1))],
)
def test_training_cli_accepts_sensor_weights(monkeypatch, cli_weights, expected):
    monkeypatch.setattr(sys, "argv", ["train", *cli_weights])
    assert experiment.parse_args().sensor_weights == expected


@pytest.mark.parametrize("values", [("0.5", "0.5", "0.1"), ("nan", "0.5", "0.5"), ("0", "0", "1")])
def test_training_cli_rejects_invalid_or_inactive_weights(monkeypatch, values):
    monkeypatch.setattr(sys, "argv", ["train", "--input-height", "6", "--sensor-weights", *values])
    with pytest.raises(SystemExit) as exc:
        experiment.parse_args()
    assert exc.value.code == 2
