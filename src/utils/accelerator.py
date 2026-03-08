from __future__ import annotations

import contextlib
import logging
import re
from typing import Iterator, Optional, Tuple

import torch
from torch import amp

_LOGGER = logging.getLogger(__name__)


def try_import_torch_npu() -> bool:
    """Best-effort import of torch_npu to register torch.npu backend."""
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return True


def _npu_available() -> bool:
    try_import_torch_npu()
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None:
        return False
    try:
        return bool(npu_mod.is_available())
    except Exception:
        return False


def _cuda_available() -> bool:
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _parse_device_spec(device: Optional[str]) -> Tuple[str, Optional[int]]:
    text = str(device or "auto").strip().lower()
    if text in {"", "auto"}:
        return "auto", None
    if text == "cpu":
        return "cpu", None
    match = re.fullmatch(r"(cuda|npu)(?::(\d+))?", text)
    if not match:
        return "auto", None
    return str(match.group(1)), int(match.group(2) or 0)


def detect_backend(preferred: Optional[str] = None) -> str:
    """
    Pick runtime backend in priority order:
    - preferred npu/cuda/cpu (if available)
    - fallback: npu > cuda > cpu
    """
    pref = str(preferred or "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "npu":
        if _npu_available():
            return "npu"
        if _cuda_available():
            _LOGGER.warning("Requested NPU but unavailable; fallback to CUDA.")
            return "cuda"
        _LOGGER.warning("Requested NPU but unavailable; fallback to CPU.")
        return "cpu"
    if pref == "cuda":
        if _cuda_available():
            return "cuda"
        if _npu_available():
            _LOGGER.warning("Requested CUDA but unavailable; fallback to NPU.")
            return "npu"
        _LOGGER.warning("Requested CUDA but unavailable; fallback to CPU.")
        return "cpu"

    if _npu_available():
        return "npu"
    if _cuda_available():
        return "cuda"
    return "cpu"


def normalize_device(device: Optional[str]) -> str:
    kind, idx = _parse_device_spec(device)
    preferred = None if kind == "auto" else kind
    backend = detect_backend(preferred)
    if backend == "cpu":
        return "cpu"
    return f"{backend}:{int(idx) if idx is not None else 0}"


def resolve_torch_device(device: Optional[str]) -> torch.device:
    normalized = normalize_device(device)
    return torch.device(normalized)


@contextlib.contextmanager
def autocast_context(device: torch.device, enabled: bool) -> Iterator[None]:
    if not enabled:
        yield
        return

    if device.type == "npu":
        try_import_torch_npu()
        npu_mod = getattr(torch, "npu", None)
        npu_amp = getattr(npu_mod, "amp", None) if npu_mod is not None else None
        npu_autocast = getattr(npu_amp, "autocast", None)
        if npu_autocast is not None:
            with npu_autocast(enabled=True):
                yield
            return

    with amp.autocast(device_type=device.type, enabled=True):
        yield
