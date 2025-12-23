from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch import amp

from vqgan import VQGAN

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TokenizationResult:
    tokens: np.ndarray  # (N, L) int64
    codebook_hw: Tuple[int, int]
    latency_sec_per_sample: float


@torch.no_grad()
def encode_windows_to_tokens(
    vqgan: VQGAN,
    windows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    desc: Optional[str] = None,
) -> TokenizationResult:
    """
    Encode window tensors into discrete codebook indices.

    Args:
        vqgan: Trained VQGAN model.
        windows: (N, 1, 12, T) float32 numpy array.
        batch_size: encoding batch size.
        device: torch device.
        use_amp: autocast on/off.
        desc: optional log prefix.
    """
    if windows.ndim != 4:
        raise ValueError(f"Expected windows as (N, C, H, W), got shape={windows.shape}")
    if windows.shape[0] == 0:
        return TokenizationResult(
            tokens=np.empty((0, 0), dtype=np.int64),
            codebook_hw=(0, 0),
            latency_sec_per_sample=0.0,
        )

    vqgan.eval()
    all_tokens = []
    codebook_hw: Optional[Tuple[int, int]] = None
    start = time.time()
    n = int(windows.shape[0])

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(windows[i : i + batch_size]).to(device=device, dtype=torch.float32, non_blocking=True)
        with amp.autocast(device_type=device.type, enabled=use_amp):
            quant_z, indices, _ = vqgan.encode(batch)
        if codebook_hw is None:
            codebook_hw = (int(quant_z.shape[2]), int(quant_z.shape[3]))
        indices = indices.view(batch.shape[0], -1).detach().to("cpu", non_blocking=False)
        all_tokens.append(indices.numpy().astype(np.int64, copy=False))

    end = time.time()
    tokens = np.concatenate(all_tokens, axis=0)
    if codebook_hw is None:
        codebook_hw = (0, 0)
    latency = float((end - start) / max(tokens.shape[0], 1))
    if desc:
        logger.info("[TOKENIZE] %s tokens=%s codebook_hw=%s latency=%.6fs/sample", desc, tokens.shape, codebook_hw, latency)

    return TokenizationResult(tokens=tokens, codebook_hw=codebook_hw, latency_sec_per_sample=latency)

