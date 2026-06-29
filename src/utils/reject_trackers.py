from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


def k_from_interrupt_time(interrupt_after_sec: float, *, window_size_sec: float, overlap: float) -> int:
    interrupt_after_sec = float(interrupt_after_sec)
    if interrupt_after_sec <= 0.0:
        return 0
    stride_sec = float(window_size_sec) * (1.0 - float(overlap))
    if stride_sec <= 0.0:
        raise ValueError(f"Invalid stride_sec={stride_sec} from window_size={window_size_sec}, overlap={overlap}")
    return max(1, int(math.ceil(interrupt_after_sec / stride_sec)))


def windows_for_delay(delay_sec: float, *, window_size_sec: float, overlap: float) -> int:
    """Number of sliding windows that span ``delay_sec`` seconds of stream time.

    Used to translate the configured authentication-result publish delay
    (``auth.result_delay_sec``) into a window count, so the server can
    accumulate that many windows before publishing one aggregated ``AuthResult``
    to the App.

    Returns ``0`` when ``delay_sec <= 0`` (meaning "no extra delay": publish one
    result per data packet, the legacy behaviour). Example: with
    ``window_size_sec=0.2`` and ``overlap=0.5`` the stride is ``0.1s`` (~10
    windows/sec), so ``delay_sec=5`` -> ``50`` windows. ``round`` is used so the
    accumulated span is the closest whole number of windows to the requested
    delay; the result is clamped to at least ``1`` window when a positive delay
    is shorter than a single stride.
    """
    delay_sec = float(delay_sec)
    if delay_sec <= 0.0:
        return 0
    stride_sec = float(window_size_sec) * (1.0 - float(overlap))
    if stride_sec <= 0.0:
        return 0
    return max(1, int(round(delay_sec / stride_sec)))


@dataclass
class ConsecutiveRejectTracker:
    windows: int = 0
    rejects: int = 0
    consecutive_rejects: int = 0
    interrupts: int = 0
    first_interrupt_window: Optional[int] = None

    def update(self, rejected: bool, *, k: int, reset_on_interrupt: bool) -> bool:
        self.windows += 1
        if rejected:
            self.rejects += 1
            self.consecutive_rejects += 1
        else:
            self.consecutive_rejects = 0

        if k <= 0:
            return False

        if self.consecutive_rejects >= k:
            self.interrupts += 1
            if self.first_interrupt_window is None:
                self.first_interrupt_window = self.windows
            if reset_on_interrupt:
                self.consecutive_rejects = 0
            return True
        return False

    def reset(self) -> None:
        self.windows = 0
        self.rejects = 0
        self.consecutive_rejects = 0
        self.interrupts = 0
        self.first_interrupt_window = None


@dataclass
class VoteRejectTracker:
    windows: int = 0
    rejects: int = 0
    interrupts: int = 0
    first_interrupt_window: Optional[int] = None
    _recent: deque[int] = field(default_factory=deque)
    _recent_rejects: int = 0

    def update(self, rejected: bool, *, window_size: int, min_rejects: int, reset_on_interrupt: bool) -> bool:
        self.windows += 1
        if rejected:
            self.rejects += 1

        window_size = int(window_size)
        min_rejects = int(min_rejects)
        if window_size <= 0 or min_rejects <= 0 or min_rejects > window_size:
            return False

        if len(self._recent) >= window_size:
            oldest = int(self._recent.popleft())
            self._recent_rejects -= oldest
        value = 1 if rejected else 0
        self._recent.append(value)
        self._recent_rejects += value

        if len(self._recent) < window_size:
            return False

        if self._recent_rejects >= min_rejects:
            self.interrupts += 1
            if self.first_interrupt_window is None:
                self.first_interrupt_window = self.windows
            if reset_on_interrupt:
                self._recent.clear()
                self._recent_rejects = 0
            return True
        return False

    @property
    def recent_windows(self) -> int:
        return int(len(self._recent))

    @property
    def recent_rejects(self) -> int:
        return int(self._recent_rejects)

    def reset(self) -> None:
        self.windows = 0
        self.rejects = 0
        self.interrupts = 0
        self.first_interrupt_window = None
        self._recent.clear()
        self._recent_rejects = 0


@dataclass
class ResultEmitGate:
    """Throttle ``AuthResult`` publication to once per accumulation period.

    The continuous per-window decision (EMA / y-of-x) is still computed for
    every window; this gate only decides *when* the latest decision is published
    to the App, so that one aggregated result covers ``target_windows`` windows
    (≈ ``auth.result_delay_sec`` seconds of stream time).

    ``target_windows <= 0`` disables throttling (the caller publishes one result
    per packet, legacy behaviour). Otherwise call :meth:`feed` once per processed
    window; it returns ``True`` exactly when the running counter crosses a
    multiple of ``target_windows`` (i.e. one delay period has just elapsed),
    carrying any remainder so the long-run cadence stays aligned with the delay.
    """

    target_windows: int = 0
    pending: int = 0

    def feed(self) -> bool:
        if int(self.target_windows) <= 0:
            return False
        self.pending += 1
        if self.pending >= int(self.target_windows):
            self.pending -= int(self.target_windows)
            return True
        return False

    def reset(self) -> None:
        self.pending = 0


@dataclass
class EMAScoreTracker:
    windows: int = 0
    interrupts: int = 0
    first_interrupt_window: Optional[int] = None
    ema_score: Optional[float] = None

    def update(self, score: float, *, alpha: float, threshold: float, reset_on_interrupt: bool) -> bool:
        self.windows += 1
        alpha_f = min(1.0, max(0.0, float(alpha)))
        score_f = float(score)
        if self.ema_score is None:
            self.ema_score = score_f
        else:
            self.ema_score = alpha_f * score_f + (1.0 - alpha_f) * float(self.ema_score)

        interrupted = bool(float(self.ema_score) < float(threshold))
        if interrupted:
            self.interrupts += 1
            if self.first_interrupt_window is None:
                self.first_interrupt_window = self.windows
            if reset_on_interrupt:
                self.ema_score = None
        return interrupted

    def reset(self) -> None:
        self.windows = 0
        self.interrupts = 0
        self.first_interrupt_window = None
        self.ema_score = None
