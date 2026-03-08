from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch

from ..utils.accelerator import autocast_context
from ..utils.ca_train import ensure_ca_train_on_path

AXIS_COLUMNS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "mag_x",
    "mag_y",
    "mag_z",
)


@dataclass(frozen=True)
class VQGANPolicy:
    user: str
    window_size: float
    overlap: float
    target_width: int
    threshold: float
    k_rejects: int
    vqgan_checkpoint: Path
    vqgan_config: Path
    vote_window_size: int = 0
    vote_min_rejects: int = 0
    model_version: str = ""


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
    args = argparse.Namespace(**cfg)
    args.use_nonlocal = bool(cfg.get("use_nonlocal", True))
    model = VQGAN(args).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def score_windows(
    model: torch.nn.Module,
    windows: np.ndarray,
    *,
    device: torch.device,
    use_amp: bool,
    score_metric: str = "mse",
    batch_size: int = 256,
) -> np.ndarray:
    if windows.size == 0:
        return np.empty((0,), dtype=np.float32)
    scores: List[np.ndarray] = []
    for i in range(0, len(windows), batch_size):
        batch_np = windows[i : i + batch_size]
        batch = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32, non_blocking=True)
        with autocast_context(device, enabled=bool(use_amp)):
            decoded, _, _ = model(batch)
            if score_metric == "l1":
                errors = torch.mean(torch.abs(batch - decoded), dim=(1, 2, 3))
            else:
                errors = torch.mean((batch - decoded) ** 2, dim=(1, 2, 3))
        scores.append((-errors).detach().cpu().numpy())
    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def windowize_dataframe(
    df,
    *,
    window_size_sec: float,
    overlap: float,
    sampling_rate_hz: int,
    target_width: int,
) -> Tuple[List[int], np.ndarray]:
    if df.empty:
        return [], np.empty((0, 1, 12, target_width), dtype=np.float32)

    window_points = max(1, int(round(window_size_sec * sampling_rate_hz)))
    step_points = max(1, int(round(window_points * (1.0 - float(overlap)))))
    values = df[list(AXIS_COLUMNS)].to_numpy(dtype=np.float32, copy=False)

    window_ids: List[int] = []
    windows: List[np.ndarray] = []
    window_id = 0
    for start in range(0, len(values) - window_points + 1, step_points):
        window_slice = values[start : start + window_points]
        acc = window_slice[:, 0:3]
        gyr = window_slice[:, 3:6]
        mag = window_slice[:, 6:9]
        acc_mag = np.linalg.norm(acc, axis=1)
        gyr_mag = np.linalg.norm(gyr, axis=1)
        mag_mag = np.linalg.norm(mag, axis=1)
        window_raw = np.vstack(
            [
                acc[:, 0],
                acc[:, 1],
                acc[:, 2],
                acc_mag,
                gyr[:, 0],
                gyr[:, 1],
                gyr[:, 2],
                gyr_mag,
                mag[:, 0],
                mag[:, 1],
                mag[:, 2],
                mag_mag,
            ]
        ).astype(np.float32, copy=False)
        if window_raw.shape[1] != target_width:
            window_raw = _resample_time_axis(window_raw, target_width)
        windows.append(window_raw[np.newaxis, :, :])
        window_ids.append(window_id)
        window_id += 1

    if not windows:
        return [], np.empty((0, 1, 12, target_width), dtype=np.float32)

    return window_ids, np.stack(windows, axis=0).astype(np.float32, copy=False)
