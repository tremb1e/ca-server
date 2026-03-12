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
