from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass(frozen=True)
class SecondaryHysteresisConfig:
    enabled: bool = True
    strategy: str = "vote"
    primary_result_interval_sec: float = 1.0
    decision_time_sec: float = 10.0
    vote_window_size: int = 10
    vote_min_rejects: int = 8
    ema_alpha: float = 0.25
    ema_reject_threshold: float = 0.8

    @property
    def normalized_primary_interval_sec(self) -> float:
        interval = float(self.primary_result_interval_sec)
        if interval <= 0.0:
            return 1.0
        return interval

    @property
    def normalized_strategy(self) -> str:
        value = str(self.strategy or "vote").strip().lower()
        if value in {"ema", "ewma"}:
            return "ema"
        return "vote"

    @property
    def ema_emit_every(self) -> int:
        interval = self.normalized_primary_interval_sec
        decision_time = float(self.decision_time_sec)
        if decision_time <= 0.0:
            return 1
        return max(1, int(math.ceil(decision_time / interval)))

    @property
    def vote_effective_window_size(self) -> int:
        window_size = int(self.vote_window_size)
        if window_size > 0:
            return window_size
        return self.ema_emit_every

    @property
    def effective_decision_time_sec(self) -> float:
        if self.normalized_strategy == "vote":
            return float(self.vote_effective_window_size * self.normalized_primary_interval_sec)
        return float(self.ema_emit_every * self.normalized_primary_interval_sec)


@dataclass(frozen=True)
class SecondaryHysteresisResult:
    ready: bool
    accept: bool
    interrupt: bool
    score: float
    threshold: float
    strategy: str
    message: str
    input_count: int
    reject_count: int
    ema_reject_score: Optional[float] = None


@dataclass
class SecondaryHysteresisTracker:
    config: SecondaryHysteresisConfig
    inputs_seen: int = 0
    decisions_emitted: int = 0
    _recent_rejects: Deque[int] = field(default_factory=deque)
    _ema_reject_score: Optional[float] = None
    _ema_since_emit: int = 0
    _ema_rejects_since_emit: int = 0

    def update(self, *, accepted: bool, interrupt: bool) -> Optional[SecondaryHysteresisResult]:
        rejected = (not bool(accepted)) or bool(interrupt)
        self.inputs_seen += 1
        if self.config.normalized_strategy == "ema":
            return self._update_ema(rejected)
        return self._update_vote(rejected)

    def _update_vote(self, rejected: bool) -> Optional[SecondaryHysteresisResult]:
        window_size = max(1, int(self.config.vote_effective_window_size))
        min_rejects = max(1, int(self.config.vote_min_rejects))
        min_rejects = min(min_rejects, window_size)

        self._recent_rejects.append(1 if rejected else 0)
        if len(self._recent_rejects) < window_size:
            return None

        reject_count = int(sum(self._recent_rejects))
        self._recent_rejects.clear()
        accepted = reject_count < min_rejects
        self.decisions_emitted += 1
        accept_count = window_size - reject_count
        score = float(accept_count / window_size)
        # Smallest accept ratio that still means reject_count < min_rejects.
        threshold = float((window_size - min_rejects + 1) / window_size)
        return SecondaryHysteresisResult(
            ready=True,
            accept=accepted,
            interrupt=not accepted,
            score=max(0.0, min(1.0, score)),
            threshold=max(0.0, min(1.0, threshold)),
            strategy="vote",
            message=(
                f"二次迟滞 vote={min_rejects}/{window_size}, "
                f"本轮不通过={reject_count}/{window_size}"
            ),
            input_count=window_size,
            reject_count=reject_count,
        )

    def _update_ema(self, rejected: bool) -> Optional[SecondaryHysteresisResult]:
        alpha = max(0.0, min(1.0, float(self.config.ema_alpha)))
        value = 1.0 if rejected else 0.0
        if self._ema_reject_score is None:
            self._ema_reject_score = value
        else:
            self._ema_reject_score = alpha * value + (1.0 - alpha) * float(self._ema_reject_score)

        self._ema_since_emit += 1
        if rejected:
            self._ema_rejects_since_emit += 1
        emit_every = int(self.config.ema_emit_every)
        if self._ema_since_emit < emit_every:
            return None

        self._ema_since_emit = 0
        reject_count = int(self._ema_rejects_since_emit)
        self._ema_rejects_since_emit = 0
        threshold_reject = max(0.0, min(1.0, float(self.config.ema_reject_threshold)))
        ema_reject = float(self._ema_reject_score)
        accepted = ema_reject < threshold_reject
        self.decisions_emitted += 1
        score = 1.0 - ema_reject
        threshold = 1.0 - threshold_reject
        return SecondaryHysteresisResult(
            ready=True,
            accept=accepted,
            interrupt=not accepted,
            score=max(0.0, min(1.0, score)),
            threshold=max(0.0, min(1.0, threshold)),
            strategy="ema",
            message=(
                f"二次迟滞 EMA={ema_reject:.6f}, alpha={alpha:.3f}, "
                f"reject_threshold={threshold_reject:.3f}, emit_every={emit_every}"
            ),
            input_count=emit_every,
            reject_count=reject_count,
            ema_reject_score=ema_reject,
        )

    def pending_count(self) -> int:
        if self.config.normalized_strategy == "ema":
            return int(self._ema_since_emit)
        return int(len(self._recent_rejects))

    def reset(self) -> None:
        self.inputs_seen = 0
        self.decisions_emitted = 0
        self._recent_rejects.clear()
        self._ema_reject_score = None
        self._ema_since_emit = 0
        self._ema_rejects_since_emit = 0
