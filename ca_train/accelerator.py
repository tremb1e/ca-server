from __future__ import annotations

import contextlib
import logging
import os
import re
from typing import Iterator, Optional, Tuple

import torch
from torch import amp

logger = logging.getLogger(__name__)


def try_import_torch_npu() -> bool:
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
    pref = str(preferred or "auto").strip().lower()
    if pref == "cpu":
        return "cpu"
    if pref == "npu":
        if _npu_available():
            return "npu"
        if _cuda_available():
            logger.warning("Requested NPU but unavailable; fallback to CUDA.")
            return "cuda"
        logger.warning("Requested NPU but unavailable; fallback to CPU.")
        return "cpu"
    if pref == "cuda":
        if _cuda_available():
            return "cuda"
        if _npu_available():
            logger.warning("Requested CUDA but unavailable; fallback to NPU.")
            return "npu"
        logger.warning("Requested CUDA but unavailable; fallback to CPU.")
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
    resolved = normalize_device(device)
    torch_device = torch.device(resolved)
    set_current_device(torch_device)
    return torch_device


def set_current_device(device: torch.device) -> None:
    if device.type == "npu":
        try_import_torch_npu()
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is not None and hasattr(npu_mod, "set_device"):
            npu_mod.set_device(device)
    elif device.type == "cuda":
        torch.cuda.set_device(device)


def device_count(backend: str) -> int:
    kind = str(backend or "").strip().lower()
    if kind == "npu":
        if not _npu_available():
            return 0
        return int(getattr(torch.npu, "device_count")())  # type: ignore[attr-defined]
    if kind == "cuda":
        if not _cuda_available():
            return 0
        return int(torch.cuda.device_count())
    return 0


def set_visible_device(backend: str, device_id: Optional[int]) -> None:
    if device_id is None:
        return
    value = str(int(device_id))
    kind = str(backend or "").strip().lower()
    if kind == "npu":
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = value
        os.environ["ASCEND_DEVICE_ID"] = "0"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif kind == "cuda":
        os.environ["CUDA_VISIBLE_DEVICES"] = value
        os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
        os.environ.pop("ASCEND_DEVICE_ID", None)


def seed_all(seed: int, *, device_type: Optional[str] = None) -> None:
    torch.manual_seed(seed)
    kind = str(device_type or "").strip().lower()
    if kind == "npu":
        try_import_torch_npu()
        npu_mod = getattr(torch, "npu", None)
        if npu_mod is not None and hasattr(npu_mod, "manual_seed_all"):
            npu_mod.manual_seed_all(seed)
    elif kind == "cuda" and _cuda_available():
        torch.cuda.manual_seed_all(seed)


def make_grad_scaler(device: torch.device, enabled: bool):
    if device.type == "npu":
        try_import_torch_npu()
        npu_mod = getattr(torch, "npu", None)
        npu_amp = getattr(npu_mod, "amp", None) if npu_mod is not None else None
        scaler_cls = getattr(npu_amp, "GradScaler", None)
        if scaler_cls is not None:
            return scaler_cls(enabled=bool(enabled))
    return amp.GradScaler(device.type, enabled=bool(enabled))


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
