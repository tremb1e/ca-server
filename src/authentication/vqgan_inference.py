from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from ca_train.reconstruction import reconstruction_errors

from ..utils.accelerator import autocast_context, set_current_device
from ..utils.ca_train import ensure_ca_train_on_path
from ..processing.magnetometer import axis_columns

AXIS_COLUMNS = axis_columns(6)  # backwards-compatible public alias


@dataclass(frozen=True)
class VQGANPolicy:
    user: str
    window_size: float
    overlap: float
    target_width: int
    threshold: float
    interrupt_rule: str
    decision_strategy: str
    k_rejects: int
    vqgan_checkpoint: Path
    vqgan_config: Path
    vote_window_size: int = 0
    vote_min_rejects: int = 0
    ema_alpha: float = 0.25
    model_version: str = ""
    input_height: int = 6
    score_metric: str = "mse"


def _resample_time_axis(window: np.ndarray, target_width: int) -> np.ndarray:
    if window.shape[1] == target_width:
        return window
    x_old = np.linspace(0.0, 1.0, window.shape[1], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_width, dtype=np.float32)
    out = np.empty((window.shape[0], target_width), dtype=np.float32)
    for i in range(window.shape[0]):
        out[i] = np.interp(x_new, x_old, window[i]).astype(np.float32, copy=False)
    return out


def load_vqgan(checkpoint: Path, *, device: torch.device, config_path: Path) -> torch.nn.Module:
    ensure_ca_train_on_path()
    from vqgan import VQGAN  # type: ignore

    if not config_path.exists():
        raise FileNotFoundError(f"Missing VQGAN config json: {config_path}")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"Unexpected VQGAN config format: {config_path}")
    input_height = int(cfg.get("input_height", 0))
    axis_columns(input_height)
    args = argparse.Namespace(**cfg)
    args.use_nonlocal = bool(cfg.get("use_nonlocal", True))
    model = VQGAN(args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


@torch.inference_mode()
def score_windows(
    model: torch.nn.Module,
    windows: np.ndarray,
    *,
    device: torch.device,
    use_amp: bool,
    score_metric: str = "mse",
    batch_size: int = 256,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if windows.size == 0:
        return np.empty((0,), dtype=np.float32)
    # NPU current-device state belongs to the executing thread; online calls
    # reach this function through asyncio.to_thread after loading on another.
    set_current_device(device)
    scores: List[np.ndarray] = []
    for i in range(0, len(windows), batch_size):
        batch_np = windows[i : i + batch_size]
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32, non_blocking=True)
        with autocast_context(device, enabled=bool(use_amp)):
            decoded, _, _ = model(batch)
        if decoded.shape != batch.shape:
            raise ValueError(f"Reconstruction shape mismatch: {decoded.shape} != {batch.shape}")
        errors = reconstruction_errors(model, batch, decoded, metric=score_metric)
        batch_scores = (-errors).detach().cpu().numpy()
        if not np.isfinite(batch_scores).all():
            raise ValueError("Model produced non-finite authentication scores")
        scores.append(batch_scores)
    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def windowize_dataframe(
    df,
    *,
    window_size_sec: float,
    overlap: float,
    sampling_rate_hz: int,
    target_width: int,
    input_height: int = 6,
) -> Tuple[List[int], np.ndarray]:
    selected_axes = axis_columns(input_height)
    if df.empty:
        return [], np.empty((0, 1, int(input_height), target_width), dtype=np.float32)

    window_points = max(1, int(round(window_size_sec * sampling_rate_hz)))
    step_points = max(1, int(round(window_points * (1.0 - float(overlap)))))
    values = df[list(selected_axes)].to_numpy(dtype=np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite sensor values for input_height={input_height}")

    window_ids: List[int] = []
    windows: List[np.ndarray] = []
    window_id = 0
    # Missing rows (including unavailable sensor values dropped by the caller)
    # must not join samples across an interruption into a synthetic window.
    breaks = np.zeros(max(0, len(values) - 1), dtype=bool)
    if "timestamp" in df.columns:
        timestamps = df["timestamp"].to_numpy(dtype=np.float64, copy=False)
        if not np.isfinite(timestamps).all():
            raise ValueError("Non-finite sensor timestamps")
        delta = np.diff(timestamps)
        breaks |= (delta <= 0) | (delta > 1.5 * 1000.0 / sampling_rate_hz)
    if "session" in df.columns:
        sessions = df["session"].to_numpy()
        breaks |= sessions[1:] != sessions[:-1]
    boundaries = np.concatenate(([0], np.flatnonzero(breaks) + 1, [len(values)]))
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        for start in range(int(left), int(right) - window_points + 1, step_points):
            window_slice = values[start : start + window_points]
            window_raw = window_slice.T.astype(np.float32, copy=True)
            if window_raw.shape[1] != target_width:
                window_raw = _resample_time_axis(window_raw, target_width)
            windows.append(window_raw[np.newaxis, :, :])
            window_ids.append(window_id)
            window_id += 1

    if not windows:
        return [], np.empty((0, 1, int(input_height), target_width), dtype=np.float32)

    return window_ids, np.stack(windows, axis=0).astype(np.float32, copy=False)
