from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

from .utils.runtime import app_root


@dataclass(frozen=True)
class ProcessingConfig:
    min_total_mb: int = 100
    target_total_mb: int = 100
    workers: int = 5
    # HMOG 攻击者用户选择（按目录名排序）：
    # - val 使用第 1~10 个用户
    # - test 使用第 11~20 个用户
    # 说明：实际选择逻辑在 processing.pipeline 中固定按上述区间切分。
    hmog_val_subject_count: int = 10
    hmog_test_subject_count: int = 10
    # HMOG 攻击者数据体量控制。
    # 默认 0=不截断（由 session 1~6 规则控制总体规模）。
    # - max_rows_per_subject: 单个 HMOG 用户最多读取多少行（跨 train/val/test 三个 CSV 合计）。
    # - max_rows_total: 单次合并（val 或 test）总共最多读取多少行（跨多个 HMOG 用户合计）。
    hmog_max_rows_per_subject: int = 0
    hmog_max_rows_total: int = 0


@dataclass(frozen=True)
class WindowConfig:
    sizes: List[float] = None  # type: ignore[assignment]
    overlap: float = 0.5
    sampling_rate_hz: int = 100

    def __post_init__(self) -> None:
        if self.sizes is None:
            object.__setattr__(self, "sizes", [0.2])


@dataclass(frozen=True)
class AuthConfig:
    max_decision_time_sec: float = 2.0
    k_rejects_mode: Literal["by_window"] = "by_window"
    # 投票规则：最近连续 N 个窗口中，若 reject>=M，则判定为恶意并触发打断。
    vote_window_size: int = 7
    vote_min_rejects: int = 6
    # 允许少量真实用户“窗口”误打断的上限（0~1）。
    target_window_frr: float = 0.10


@dataclass(frozen=True)
class CAConfig:
    processing: ProcessingConfig = ProcessingConfig()
    windows: WindowConfig = WindowConfig()
    auth: AuthConfig = AuthConfig()

    def k_rejects_for_window(self, window_size_sec: float) -> int:
        """Compute consecutive rejects K for a given window size.

        Policy: convert decision time T into K using stride = window_size * (1 - overlap),
        so that the interrupt is not earlier than T seconds.
        """
        window_size_sec = float(window_size_sec)
        if window_size_sec <= 0.0:
            raise ValueError(f"window_size_sec must be > 0, got {window_size_sec}")
        max_t = float(self.auth.max_decision_time_sec)
        if max_t <= 0.0:
            return 0
        stride_sec = window_size_sec * (1.0 - float(self.windows.overlap))
        if stride_sec <= 0.0:
            raise ValueError(f"Invalid stride_sec={stride_sec} from window_size={window_size_sec}, overlap={self.windows.overlap}")
        return max(1, int(math.ceil(max_t / stride_sec)))


def _default_config_path() -> Path:
    configured = os.getenv("CA_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return app_root() / "ca_config.toml"


def load_ca_config(path: Optional[Path] = None) -> CAConfig:
    path = Path(path) if path is not None else _default_config_path()
    if not path.exists():
        return CAConfig()

    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    proc_raw = raw.get("processing", {}) or {}
    win_raw = raw.get("windows", {}) or {}
    auth_raw = raw.get("auth", {}) or {}

    processing = ProcessingConfig(
        min_total_mb=int(proc_raw.get("min_total_mb", ProcessingConfig.min_total_mb)),
        target_total_mb=int(proc_raw.get("target_total_mb", ProcessingConfig.target_total_mb)),
        workers=int(proc_raw.get("workers", ProcessingConfig.workers)),
        hmog_val_subject_count=int(proc_raw.get("hmog_val_subject_count", ProcessingConfig.hmog_val_subject_count)),
        hmog_test_subject_count=int(proc_raw.get("hmog_test_subject_count", ProcessingConfig.hmog_test_subject_count)),
        hmog_max_rows_per_subject=int(proc_raw.get("hmog_max_rows_per_subject", ProcessingConfig.hmog_max_rows_per_subject)),
        hmog_max_rows_total=int(proc_raw.get("hmog_max_rows_total", ProcessingConfig.hmog_max_rows_total)),
    )

    sizes = win_raw.get("sizes", None)
    if sizes is not None:
        sizes = [float(x) for x in sizes]
        sizes = sorted({round(float(x), 3) for x in sizes})
    windows = WindowConfig(
        sizes=sizes,  # type: ignore[arg-type]
        overlap=float(win_raw.get("overlap", WindowConfig.overlap)),
        sampling_rate_hz=int(win_raw.get("sampling_rate_hz", WindowConfig.sampling_rate_hz)),
    )

    target_window_frr_raw = auth_raw.get("target_window_frr", None)
    if target_window_frr_raw is None:
        # Backward-compatible fallback
        target_window_frr_raw = auth_raw.get("target_session_frr", AuthConfig.target_window_frr)

    auth = AuthConfig(
        max_decision_time_sec=float(auth_raw.get("max_decision_time_sec", AuthConfig.max_decision_time_sec)),
        k_rejects_mode=str(auth_raw.get("k_rejects_mode", AuthConfig.k_rejects_mode)),  # type: ignore[arg-type]
        vote_window_size=int(auth_raw.get("vote_window_size", AuthConfig.vote_window_size)),
        vote_min_rejects=int(auth_raw.get("vote_min_rejects", AuthConfig.vote_min_rejects)),
        target_window_frr=float(target_window_frr_raw),
    )

    return CAConfig(processing=processing, windows=windows, auth=auth)


_CACHED: Optional[CAConfig] = None


def get_ca_config(path: Optional[Path] = None) -> CAConfig:
    global _CACHED
    if _CACHED is None:
        _CACHED = load_ca_config(path)
    return _CACHED
