"""Shared sensor reconstruction scoring for training and authentication."""

from __future__ import annotations

import math
from numbers import Real
from typing import Optional, Sequence, Tuple


# Order is accelerometer, gyroscope, magnetometer.  The value is deliberately
# kept in one module so training and every inference path use the same rule.
DEFAULT_SENSOR_WEIGHTS = (0.5, 0.5, 0.0)
SENSOR_INPUT_MASKING_VERSION = 1


def validate_sensor_weights(weights: Optional[Sequence[float]]) -> Optional[Tuple[float, ...]]:
    """Validate and normalize the configured modality weights representation."""
    if weights is None:
        return None
    message = "sensor_weights must contain three finite nonnegative numbers summing to 1"
    if not isinstance(weights, (list, tuple)) or any(
        isinstance(value, bool) or not isinstance(value, Real) for value in weights
    ):
        raise ValueError(message)
    values = tuple(float(value) for value in weights)
    if (
        len(values) != 3
        or any(not math.isfinite(value) or value < 0 for value in values)
        or not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-6)
    ):
        raise ValueError(message)
    return values


def mask_inactive_sensor_inputs(batch, weights: Optional[Sequence[float]]):
    """Keep zero-weight sensor groups out of the encoder as well as the loss."""
    weights = validate_sensor_weights(weights)
    if weights is None or all(weight > 0 for weight in weights):
        return batch
    if batch.ndim != 4 or int(batch.shape[2]) not in (6, 9):
        raise ValueError("Weighted sensor input requires 6 or 9 sensor axes")
    inactive = [index for index, weight in enumerate(weights[: int(batch.shape[2]) // 3]) if weight == 0]
    if not inactive:
        return batch
    masked = batch.clone()
    for index in inactive:
        masked[:, :, index * 3 : (index + 1) * 3, :] = 0
    return masked


def reconstruction_errors(model, batch, decoded, *, metric: str = "mse"):
    """Return one FP32 reconstruction error per NCHW window.

    A model config without ``sensor_weights`` is a legacy checkpoint and keeps
    the historical all-axis mean score.  New checkpoints bind their weights to
    the model, which keeps training, policy search, and online authentication
    on the same score scale.
    """
    if metric not in {"mse", "l1"}:
        raise ValueError(f"Unsupported reconstruction metric: {metric}")
    if batch.ndim != 4 or decoded.shape != batch.shape:
        raise ValueError("Reconstruction requires matching NCHW tensors")
    difference = batch.float() - decoded.float()
    errors = difference.square() if metric == "mse" else difference.abs()
    weights = validate_sensor_weights(getattr(getattr(model, "module", model), "sensor_weights", None))
    if weights is None:
        return errors.mean(dim=(1, 2, 3))

    height = int(batch.shape[2])
    if height not in (6, 9):
        raise ValueError("Weighted reconstruction requires 6 or 9 sensor axes")
    active_weights = weights[: height // 3]
    total_weight = sum(active_weights)
    if total_weight <= 0:
        raise ValueError("Active sensor weights must have a positive sum")
    # Each modality contributes its mean over its three axes; 6-axis models
    # renormalize the active accelerometer/gyroscope weights.
    return sum(
        errors[:, :, index * 3 : (index + 1) * 3, :].mean(dim=(1, 2, 3))
        * (weight / total_weight)
        for index, weight in enumerate(active_weights)
    )
