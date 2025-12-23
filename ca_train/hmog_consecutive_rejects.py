from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def k_from_interrupt_time(interrupt_after_sec: float, *, window_size_sec: float, overlap: float) -> int:
    """
    Convert a fixed interrupt time (seconds) into an equivalent "K consecutive rejects"
    count, using stride = window_size * (1 - overlap).

    We use ceil() so that the effective interrupt time is **not earlier** than the
    requested time, which is more usability-friendly.
    """
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
    """
    Sliding vote interrupt rule tracker.

    Rule:
      - reject_window = (score < threshold)
      - maintain the last `window_size` reject decisions
      - trigger an interrupt when `rejects_in_last_window >= min_rejects`
    """

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
        if window_size <= 0 or min_rejects <= 0:
            return False
        if min_rejects > window_size:
            return False

        # Maintain fixed-size queue without relying on deque(maxlen) so we can
        # track the reject count efficiently.
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


def compute_k_consecutive_reject_session_metrics(
    scored_windows: Iterable[Tuple[str, str, int, float]],
    *,
    threshold: float,
    k: int,
    reset_on_interrupt: bool = True,
) -> Dict[str, float]:
    """
    Compute sequence/session-level metrics under the rule:
      - reject_window = (score < threshold)
      - interrupt occurs when there are k consecutive rejects within a session

    The input must be ordered such that windows belonging to the same (subject, session)
    are contiguous and in chronological order (as in the server CSVs).

    Args:
        scored_windows: iterable of (subject, session, label, score) tuples.
          - label: 1=genuine, 0=impostor.
          - score: larger => more genuine.
        threshold: decision threshold on score.
        k: consecutive reject count needed to trigger an interrupt.
        reset_on_interrupt: if True, reset the consecutive counter after a trigger
          (allows counting multiple interrupts per session).
    """
    k = int(k)

    genuine_sessions = 0
    impostor_sessions = 0

    genuine_interrupted_sessions = 0
    impostor_detected_sessions = 0

    genuine_total_interrupts = 0
    impostor_total_interrupts = 0

    genuine_total_windows = 0
    impostor_total_windows = 0

    genuine_total_reject_windows = 0
    impostor_total_reject_windows = 0

    genuine_first_interrupt_sum = 0
    impostor_first_interrupt_sum = 0
    genuine_first_interrupt_n = 0
    impostor_first_interrupt_n = 0

    cur_key: Optional[Tuple[str, str]] = None
    cur_label: Optional[int] = None
    cur_state = ConsecutiveRejectTracker()

    def _flush_current() -> None:
        nonlocal genuine_sessions
        nonlocal impostor_sessions
        nonlocal genuine_interrupted_sessions
        nonlocal impostor_detected_sessions
        nonlocal genuine_total_interrupts
        nonlocal impostor_total_interrupts
        nonlocal genuine_total_windows
        nonlocal impostor_total_windows
        nonlocal genuine_total_reject_windows
        nonlocal impostor_total_reject_windows
        nonlocal genuine_first_interrupt_sum
        nonlocal impostor_first_interrupt_sum
        nonlocal genuine_first_interrupt_n
        nonlocal impostor_first_interrupt_n
        nonlocal cur_key, cur_label, cur_state

        if cur_key is None or cur_label is None:
            return

        interrupted = int(cur_state.interrupts > 0)
        first_interrupt = int(cur_state.first_interrupt_window or 0)

        if int(cur_label) == 1:
            genuine_sessions += 1
            genuine_total_windows += int(cur_state.windows)
            genuine_total_reject_windows += int(cur_state.rejects)
            genuine_total_interrupts += int(cur_state.interrupts)
            genuine_interrupted_sessions += interrupted
            if interrupted:
                genuine_first_interrupt_sum += first_interrupt
                genuine_first_interrupt_n += 1
        else:
            impostor_sessions += 1
            impostor_total_windows += int(cur_state.windows)
            impostor_total_reject_windows += int(cur_state.rejects)
            impostor_total_interrupts += int(cur_state.interrupts)
            impostor_detected_sessions += interrupted
            if interrupted:
                impostor_first_interrupt_sum += first_interrupt
                impostor_first_interrupt_n += 1

        cur_key = None
        cur_label = None
        cur_state = ConsecutiveRejectTracker()

    for subject, session, label, score in scored_windows:
        key = (str(subject), str(session))
        label_int = int(label)
        score_f = float(score)

        if cur_key is None:
            cur_key = key
            cur_label = label_int
        elif key != cur_key:
            _flush_current()
            cur_key = key
            cur_label = label_int
        elif cur_label is not None and label_int != int(cur_label):
            # Session keys should be homogeneous; if not, treat as a boundary.
            _flush_current()
            cur_key = key
            cur_label = label_int

        rejected = bool(score_f < float(threshold))
        cur_state.update(rejected, k=k, reset_on_interrupt=bool(reset_on_interrupt))

    _flush_current()

    session_frr = float(genuine_interrupted_sessions / genuine_sessions) if genuine_sessions else 0.0
    session_tpr = float(impostor_detected_sessions / impostor_sessions) if impostor_sessions else 0.0
    session_far = float((impostor_sessions - impostor_detected_sessions) / impostor_sessions) if impostor_sessions else 0.0

    genuine_avg_interrupts = float(genuine_total_interrupts / genuine_sessions) if genuine_sessions else 0.0
    impostor_avg_interrupts = float(impostor_total_interrupts / impostor_sessions) if impostor_sessions else 0.0

    genuine_reject_rate = float(genuine_total_reject_windows / genuine_total_windows) if genuine_total_windows else 0.0
    impostor_reject_rate = float(impostor_total_reject_windows / impostor_total_windows) if impostor_total_windows else 0.0

    genuine_mean_first_interrupt = (
        float(genuine_first_interrupt_sum / genuine_first_interrupt_n) if genuine_first_interrupt_n else 0.0
    )
    impostor_mean_first_interrupt = (
        float(impostor_first_interrupt_sum / impostor_first_interrupt_n) if impostor_first_interrupt_n else 0.0
    )

    return {
        "k_rejects": float(k),
        "threshold": float(threshold),
        "session_far": session_far,
        "session_frr": session_frr,
        "session_tpr": session_tpr,
        "genuine_sessions": float(genuine_sessions),
        "impostor_sessions": float(impostor_sessions),
        "genuine_interrupted_sessions": float(genuine_interrupted_sessions),
        "impostor_detected_sessions": float(impostor_detected_sessions),
        "genuine_total_windows": float(genuine_total_windows),
        "impostor_total_windows": float(impostor_total_windows),
        "genuine_total_reject_windows": float(genuine_total_reject_windows),
        "impostor_total_reject_windows": float(impostor_total_reject_windows),
        "genuine_window_reject_rate": genuine_reject_rate,
        "impostor_window_reject_rate": impostor_reject_rate,
        "genuine_total_interrupts": float(genuine_total_interrupts),
        "impostor_total_interrupts": float(impostor_total_interrupts),
        "genuine_avg_interrupts_per_session": genuine_avg_interrupts,
        "impostor_avg_interrupts_per_session": impostor_avg_interrupts,
        "genuine_mean_first_interrupt_window": genuine_mean_first_interrupt,
        "impostor_mean_first_interrupt_window": impostor_mean_first_interrupt,
    }


def compute_vote_reject_session_metrics(
    scored_windows: Iterable[Tuple[str, str, int, float]],
    *,
    threshold: float,
    window_size: int,
    min_rejects: int,
    reset_on_interrupt: bool = True,
) -> Dict[str, float]:
    """
    Compute sequence/session-level metrics under the rule:
      - reject_window = (score < threshold)
      - interrupt occurs when, within the most recent `window_size` windows,
        the number of reject windows is >= `min_rejects`

    The input must be ordered such that windows belonging to the same (subject, session)
    are contiguous and in chronological order (as in the server CSVs).
    """
    window_size = int(window_size)
    min_rejects = int(min_rejects)

    genuine_sessions = 0
    impostor_sessions = 0

    genuine_interrupted_sessions = 0
    impostor_detected_sessions = 0

    genuine_total_interrupts = 0
    impostor_total_interrupts = 0

    genuine_total_windows = 0
    impostor_total_windows = 0

    genuine_total_reject_windows = 0
    impostor_total_reject_windows = 0

    genuine_first_interrupt_sum = 0
    impostor_first_interrupt_sum = 0
    genuine_first_interrupt_n = 0
    impostor_first_interrupt_n = 0

    cur_key: Optional[Tuple[str, str]] = None
    cur_label: Optional[int] = None
    cur_state = VoteRejectTracker()

    def _flush_current() -> None:
        nonlocal genuine_sessions
        nonlocal impostor_sessions
        nonlocal genuine_interrupted_sessions
        nonlocal impostor_detected_sessions
        nonlocal genuine_total_interrupts
        nonlocal impostor_total_interrupts
        nonlocal genuine_total_windows
        nonlocal impostor_total_windows
        nonlocal genuine_total_reject_windows
        nonlocal impostor_total_reject_windows
        nonlocal genuine_first_interrupt_sum
        nonlocal impostor_first_interrupt_sum
        nonlocal genuine_first_interrupt_n
        nonlocal impostor_first_interrupt_n
        nonlocal cur_key, cur_label, cur_state

        if cur_key is None or cur_label is None:
            return

        interrupted = int(cur_state.interrupts > 0)
        first_interrupt = int(cur_state.first_interrupt_window or 0)

        if int(cur_label) == 1:
            genuine_sessions += 1
            genuine_total_windows += int(cur_state.windows)
            genuine_total_reject_windows += int(cur_state.rejects)
            genuine_total_interrupts += int(cur_state.interrupts)
            genuine_interrupted_sessions += interrupted
            if interrupted:
                genuine_first_interrupt_sum += first_interrupt
                genuine_first_interrupt_n += 1
        else:
            impostor_sessions += 1
            impostor_total_windows += int(cur_state.windows)
            impostor_total_reject_windows += int(cur_state.rejects)
            impostor_total_interrupts += int(cur_state.interrupts)
            impostor_detected_sessions += interrupted
            if interrupted:
                impostor_first_interrupt_sum += first_interrupt
                impostor_first_interrupt_n += 1

        cur_key = None
        cur_label = None
        cur_state = VoteRejectTracker()

    threshold = float(threshold)
    for subject, session, label, score in scored_windows:
        key = (str(subject), str(session))
        label_int = int(label)
        score_f = float(score)

        if cur_key is None:
            cur_key = key
            cur_label = label_int
        elif key != cur_key:
            _flush_current()
            cur_key = key
            cur_label = label_int
        elif cur_label is not None and label_int != int(cur_label):
            _flush_current()
            cur_key = key
            cur_label = label_int

        rejected = bool(score_f < threshold)
        cur_state.update(
            rejected,
            window_size=window_size,
            min_rejects=min_rejects,
            reset_on_interrupt=bool(reset_on_interrupt),
        )

    _flush_current()

    session_frr = float(genuine_interrupted_sessions / genuine_sessions) if genuine_sessions else 0.0
    session_tpr = float(impostor_detected_sessions / impostor_sessions) if impostor_sessions else 0.0
    session_far = float((impostor_sessions - impostor_detected_sessions) / impostor_sessions) if impostor_sessions else 0.0

    genuine_avg_interrupts = float(genuine_total_interrupts / genuine_sessions) if genuine_sessions else 0.0
    impostor_avg_interrupts = float(impostor_total_interrupts / impostor_sessions) if impostor_sessions else 0.0

    genuine_reject_rate = float(genuine_total_reject_windows / genuine_total_windows) if genuine_total_windows else 0.0
    impostor_reject_rate = float(impostor_total_reject_windows / impostor_total_windows) if impostor_total_windows else 0.0

    genuine_mean_first_interrupt = (
        float(genuine_first_interrupt_sum / genuine_first_interrupt_n) if genuine_first_interrupt_n else 0.0
    )
    impostor_mean_first_interrupt = (
        float(impostor_first_interrupt_sum / impostor_first_interrupt_n) if impostor_first_interrupt_n else 0.0
    )

    return {
        "vote_window_size": float(window_size),
        "vote_min_rejects": float(min_rejects),
        "threshold": float(threshold),
        "session_far": session_far,
        "session_frr": session_frr,
        "session_tpr": session_tpr,
        "genuine_sessions": float(genuine_sessions),
        "impostor_sessions": float(impostor_sessions),
        "genuine_interrupted_sessions": float(genuine_interrupted_sessions),
        "impostor_detected_sessions": float(impostor_detected_sessions),
        "genuine_total_windows": float(genuine_total_windows),
        "impostor_total_windows": float(impostor_total_windows),
        "genuine_total_reject_windows": float(genuine_total_reject_windows),
        "impostor_total_reject_windows": float(impostor_total_reject_windows),
        "genuine_window_reject_rate": genuine_reject_rate,
        "impostor_window_reject_rate": impostor_reject_rate,
        "genuine_total_interrupts": float(genuine_total_interrupts),
        "impostor_total_interrupts": float(impostor_total_interrupts),
        "genuine_avg_interrupts_per_session": genuine_avg_interrupts,
        "impostor_avg_interrupts_per_session": impostor_avg_interrupts,
        "genuine_mean_first_interrupt_window": genuine_mean_first_interrupt,
        "impostor_mean_first_interrupt_window": impostor_mean_first_interrupt,
    }


def compute_genuine_session_thresholds_no_interrupt(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    k: int,
) -> List[float]:
    """
    Compute, for each *genuine* session, the maximum threshold that guarantees
    **no interrupt** under the rule "interrupt when there are k consecutive rejects",
    assuming reject_window = (score < threshold).

    For a single session with score sequence s_1..s_n, no k-consecutive-rejects is
    equivalent to: for every i, max(s_i..s_{i+k-1}) >= threshold.
    So the largest safe threshold is:
        threshold_session = min_i max(s_i..s_{i+k-1})
    (and +inf if n < k).

    Args:
        session_ids: per-window session id, must be contiguous per session.
        labels: per-window label (1=genuine, 0=impostor), constant within a session.
        scores: per-window score (larger = more genuine).
        k: consecutive reject count.
    """
    k = int(k)
    if k <= 0:
        return []

    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    # Only genuine sessions matter for session_FRR; filtering avoids O(N*k) work
    # on massive impostor windows when k is large (e.g., small window sizes).
    genuine_mask = labels_arr == 1
    if not np.any(genuine_mask):
        return []
    session_ids_arr = session_ids_arr[genuine_mask]
    scores_arr = scores_arr[genuine_mask]

    thresholds: List[float] = []
    cur_sid: Optional[int] = None
    dq: deque[float] = deque(maxlen=k)
    min_window_max = float("inf")

    def _flush() -> None:
        nonlocal cur_sid, dq, min_window_max, thresholds
        if cur_sid is None:
            return
        thresholds.append(float(min_window_max))
        cur_sid = None
        dq.clear()
        min_window_max = float("inf")

    for sid, score in zip(session_ids_arr.tolist(), scores_arr.tolist()):
        sid_i = int(sid)
        score_f = float(score)

        if cur_sid is None:
            cur_sid = sid_i
        elif sid_i != int(cur_sid):
            _flush()
            cur_sid = sid_i

        dq.append(score_f)
        if len(dq) == k:
            window_max = float(max(dq))
            if window_max < min_window_max:
                min_window_max = window_max

    _flush()
    return thresholds


def compute_genuine_session_thresholds_no_interrupt_vote(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    window_size: int,
    min_rejects: int,
) -> List[float]:
    """
    Compute, for each *genuine* session, the maximum threshold that guarantees
    **no interrupt** under the rule:
      - interrupt when, within the most recent `window_size` windows,
        rejects >= `min_rejects`
    assuming reject_window = (score < threshold).

    For a single length-N window, rejects >= min_rejects iff threshold > s_m
    where s_m is the `min_rejects`-th smallest score in that window.
    So the largest safe threshold for the session is:
        threshold_session = min_over_windows( mth_smallest(window_scores) )
    (and +inf if session_length < window_size).
    """
    window_size = int(window_size)
    min_rejects = int(min_rejects)
    if window_size <= 0 or min_rejects <= 0:
        return []

    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    genuine_mask = labels_arr == 1
    if not np.any(genuine_mask):
        return []
    session_ids_arr = session_ids_arr[genuine_mask]
    scores_arr = scores_arr[genuine_mask]

    thresholds: List[float] = []
    cur_sid: Optional[int] = None
    dq: deque[float] = deque(maxlen=window_size)
    min_window_mth = float("inf")

    def _flush() -> None:
        nonlocal cur_sid, dq, min_window_mth, thresholds
        if cur_sid is None:
            return
        thresholds.append(float(min_window_mth))
        cur_sid = None
        dq.clear()
        min_window_mth = float("inf")

    if min_rejects > window_size:
        # Impossible to trigger -> any threshold is safe.
        for sid in session_ids_arr.tolist():
            sid_i = int(sid)
            if cur_sid is None:
                cur_sid = sid_i
            elif sid_i != int(cur_sid):
                _flush()
                cur_sid = sid_i
        _flush()
        return [float("inf")] * len(thresholds) if thresholds else []

    for sid, score in zip(session_ids_arr.tolist(), scores_arr.tolist()):
        sid_i = int(sid)
        score_f = float(score)

        if cur_sid is None:
            cur_sid = sid_i
        elif sid_i != int(cur_sid):
            _flush()
            cur_sid = sid_i

        dq.append(score_f)
        if len(dq) == window_size:
            mth = float(sorted(dq)[min_rejects - 1])
            if mth < min_window_mth:
                min_window_mth = mth

    _flush()
    return thresholds


def select_threshold_by_k_session_frr_from_genuine_thresholds(
    genuine_session_thresholds: Sequence[float],
    *,
    target_session_frr: float,
) -> float:
    """
    Select the *highest* threshold such that session_FRR <= target_session_frr,
    using only the per-genuine-session "no-interrupt" thresholds.

    A genuine session is interrupted iff threshold > threshold_session.
    So session_FRR(threshold) = P(threshold_session < threshold).
    """
    if not genuine_session_thresholds:
        raise ValueError("Empty genuine_session_thresholds")

    target_session_frr = float(target_session_frr)
    if target_session_frr < 0.0:
        target_session_frr = 0.0
    if target_session_frr > 1.0:
        target_session_frr = 1.0

    finite = [float(t) for t in genuine_session_thresholds if math.isfinite(float(t))]
    if not finite:
        # No session had length >= k; any threshold is safe in terms of interrupts.
        return float("inf")

    finite.sort()
    n = len(finite)
    allowed = int(math.floor(target_session_frr * n))
    idx = min(max(allowed, 0), n - 1)
    return float(finite[idx])


def compute_k_consecutive_reject_session_metrics_from_arrays(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    k: int,
    reset_on_interrupt: bool = True,
) -> Dict[str, float]:
    """
    Same as `compute_k_consecutive_reject_session_metrics`, but works directly on
    per-window arrays and uses an integer session id (contiguous by session).

    Args:
        session_ids: per-window session id, must be contiguous per session.
        labels: 1=genuine, 0=impostor.
        scores: larger => more genuine.
        threshold: accept if score >= threshold.
        k: consecutive rejects to trigger an interrupt.
    """
    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    scored_iter = (
        (int(sid), int(label), float(score))
        for sid, label, score in zip(session_ids_arr.tolist(), labels_arr.tolist(), scores_arr.tolist())
    )

    k = int(k)
    threshold = float(threshold)
    reset_on_interrupt = bool(reset_on_interrupt)

    genuine_sessions = 0
    impostor_sessions = 0

    genuine_interrupted_sessions = 0
    impostor_detected_sessions = 0

    genuine_total_interrupts = 0
    impostor_total_interrupts = 0

    genuine_total_windows = 0
    impostor_total_windows = 0

    genuine_total_reject_windows = 0
    impostor_total_reject_windows = 0

    genuine_first_interrupt_sum = 0
    impostor_first_interrupt_sum = 0
    genuine_first_interrupt_n = 0
    impostor_first_interrupt_n = 0

    cur_sid: Optional[int] = None
    cur_label: Optional[int] = None
    cur_state = ConsecutiveRejectTracker()

    def _flush_current() -> None:
        nonlocal genuine_sessions
        nonlocal impostor_sessions
        nonlocal genuine_interrupted_sessions
        nonlocal impostor_detected_sessions
        nonlocal genuine_total_interrupts
        nonlocal impostor_total_interrupts
        nonlocal genuine_total_windows
        nonlocal impostor_total_windows
        nonlocal genuine_total_reject_windows
        nonlocal impostor_total_reject_windows
        nonlocal genuine_first_interrupt_sum
        nonlocal impostor_first_interrupt_sum
        nonlocal genuine_first_interrupt_n
        nonlocal impostor_first_interrupt_n
        nonlocal cur_sid, cur_label, cur_state

        if cur_sid is None or cur_label is None:
            return

        interrupted = int(cur_state.interrupts > 0)
        first_interrupt = int(cur_state.first_interrupt_window or 0)

        if int(cur_label) == 1:
            genuine_sessions += 1
            genuine_total_windows += int(cur_state.windows)
            genuine_total_reject_windows += int(cur_state.rejects)
            genuine_total_interrupts += int(cur_state.interrupts)
            genuine_interrupted_sessions += interrupted
            if interrupted:
                genuine_first_interrupt_sum += first_interrupt
                genuine_first_interrupt_n += 1
        else:
            impostor_sessions += 1
            impostor_total_windows += int(cur_state.windows)
            impostor_total_reject_windows += int(cur_state.rejects)
            impostor_total_interrupts += int(cur_state.interrupts)
            impostor_detected_sessions += interrupted
            if interrupted:
                impostor_first_interrupt_sum += first_interrupt
                impostor_first_interrupt_n += 1

        cur_sid = None
        cur_label = None
        cur_state = ConsecutiveRejectTracker()

    for sid, label, score in scored_iter:
        sid_i = int(sid)
        label_i = int(label)
        score_f = float(score)

        if cur_sid is None:
            cur_sid = sid_i
            cur_label = label_i
        elif sid_i != int(cur_sid):
            _flush_current()
            cur_sid = sid_i
            cur_label = label_i
        elif cur_label is not None and label_i != int(cur_label):
            _flush_current()
            cur_sid = sid_i
            cur_label = label_i

        rejected = bool(score_f < threshold)
        cur_state.update(rejected, k=k, reset_on_interrupt=reset_on_interrupt)

    _flush_current()

    session_frr = float(genuine_interrupted_sessions / genuine_sessions) if genuine_sessions else 0.0
    session_tpr = float(impostor_detected_sessions / impostor_sessions) if impostor_sessions else 0.0
    session_far = float((impostor_sessions - impostor_detected_sessions) / impostor_sessions) if impostor_sessions else 0.0

    genuine_avg_interrupts = float(genuine_total_interrupts / genuine_sessions) if genuine_sessions else 0.0
    impostor_avg_interrupts = float(impostor_total_interrupts / impostor_sessions) if impostor_sessions else 0.0

    genuine_reject_rate = float(genuine_total_reject_windows / genuine_total_windows) if genuine_total_windows else 0.0
    impostor_reject_rate = float(impostor_total_reject_windows / impostor_total_windows) if impostor_total_windows else 0.0

    genuine_mean_first_interrupt = (
        float(genuine_first_interrupt_sum / genuine_first_interrupt_n) if genuine_first_interrupt_n else 0.0
    )
    impostor_mean_first_interrupt = (
        float(impostor_first_interrupt_sum / impostor_first_interrupt_n) if impostor_first_interrupt_n else 0.0
    )

    return {
        "k_rejects": float(k),
        "threshold": float(threshold),
        "session_far": session_far,
        "session_frr": session_frr,
        "session_tpr": session_tpr,
        "genuine_sessions": float(genuine_sessions),
        "impostor_sessions": float(impostor_sessions),
        "genuine_interrupted_sessions": float(genuine_interrupted_sessions),
        "impostor_detected_sessions": float(impostor_detected_sessions),
        "genuine_total_windows": float(genuine_total_windows),
        "impostor_total_windows": float(impostor_total_windows),
        "genuine_total_reject_windows": float(genuine_total_reject_windows),
        "impostor_total_reject_windows": float(impostor_total_reject_windows),
        "genuine_window_reject_rate": genuine_reject_rate,
        "impostor_window_reject_rate": impostor_reject_rate,
        "genuine_total_interrupts": float(genuine_total_interrupts),
        "impostor_total_interrupts": float(impostor_total_interrupts),
        "genuine_avg_interrupts_per_session": genuine_avg_interrupts,
        "impostor_avg_interrupts_per_session": impostor_avg_interrupts,
        "genuine_mean_first_interrupt_window": genuine_mean_first_interrupt,
        "impostor_mean_first_interrupt_window": impostor_mean_first_interrupt,
    }


def compute_vote_reject_session_metrics_from_arrays(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    window_size: int,
    min_rejects: int,
    reset_on_interrupt: bool = True,
) -> Dict[str, float]:
    """
    Same as `compute_vote_reject_session_metrics`, but works directly on
    per-window arrays and uses an integer session id (contiguous by session).
    """
    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    window_size = int(window_size)
    min_rejects = int(min_rejects)
    threshold = float(threshold)
    reset_on_interrupt = bool(reset_on_interrupt)

    genuine_sessions = 0
    impostor_sessions = 0

    genuine_interrupted_sessions = 0
    impostor_detected_sessions = 0

    genuine_total_interrupts = 0
    impostor_total_interrupts = 0

    genuine_total_windows = 0
    impostor_total_windows = 0

    genuine_total_reject_windows = 0
    impostor_total_reject_windows = 0

    genuine_first_interrupt_sum = 0
    impostor_first_interrupt_sum = 0
    genuine_first_interrupt_n = 0
    impostor_first_interrupt_n = 0

    cur_sid: Optional[int] = None
    cur_label: Optional[int] = None
    cur_state = VoteRejectTracker()

    def _flush_current() -> None:
        nonlocal genuine_sessions
        nonlocal impostor_sessions
        nonlocal genuine_interrupted_sessions
        nonlocal impostor_detected_sessions
        nonlocal genuine_total_interrupts
        nonlocal impostor_total_interrupts
        nonlocal genuine_total_windows
        nonlocal impostor_total_windows
        nonlocal genuine_total_reject_windows
        nonlocal impostor_total_reject_windows
        nonlocal genuine_first_interrupt_sum
        nonlocal impostor_first_interrupt_sum
        nonlocal genuine_first_interrupt_n
        nonlocal impostor_first_interrupt_n
        nonlocal cur_sid, cur_label, cur_state

        if cur_sid is None or cur_label is None:
            return

        interrupted = int(cur_state.interrupts > 0)
        first_interrupt = int(cur_state.first_interrupt_window or 0)

        if int(cur_label) == 1:
            genuine_sessions += 1
            genuine_total_windows += int(cur_state.windows)
            genuine_total_reject_windows += int(cur_state.rejects)
            genuine_total_interrupts += int(cur_state.interrupts)
            genuine_interrupted_sessions += interrupted
            if interrupted:
                genuine_first_interrupt_sum += first_interrupt
                genuine_first_interrupt_n += 1
        else:
            impostor_sessions += 1
            impostor_total_windows += int(cur_state.windows)
            impostor_total_reject_windows += int(cur_state.rejects)
            impostor_total_interrupts += int(cur_state.interrupts)
            impostor_detected_sessions += interrupted
            if interrupted:
                impostor_first_interrupt_sum += first_interrupt
                impostor_first_interrupt_n += 1

        cur_sid = None
        cur_label = None
        cur_state = VoteRejectTracker()

    for sid, label, score in zip(session_ids_arr.tolist(), labels_arr.tolist(), scores_arr.tolist()):
        sid_i = int(sid)
        label_i = int(label)
        score_f = float(score)

        if cur_sid is None:
            cur_sid = sid_i
            cur_label = label_i
        elif sid_i != int(cur_sid):
            _flush_current()
            cur_sid = sid_i
            cur_label = label_i
        elif cur_label is not None and label_i != int(cur_label):
            _flush_current()
            cur_sid = sid_i
            cur_label = label_i

        rejected = bool(score_f < threshold)
        cur_state.update(
            rejected,
            window_size=window_size,
            min_rejects=min_rejects,
            reset_on_interrupt=reset_on_interrupt,
        )

    _flush_current()

    session_frr = float(genuine_interrupted_sessions / genuine_sessions) if genuine_sessions else 0.0
    session_tpr = float(impostor_detected_sessions / impostor_sessions) if impostor_sessions else 0.0
    session_far = float((impostor_sessions - impostor_detected_sessions) / impostor_sessions) if impostor_sessions else 0.0

    genuine_avg_interrupts = float(genuine_total_interrupts / genuine_sessions) if genuine_sessions else 0.0
    impostor_avg_interrupts = float(impostor_total_interrupts / impostor_sessions) if impostor_sessions else 0.0

    genuine_reject_rate = float(genuine_total_reject_windows / genuine_total_windows) if genuine_total_windows else 0.0
    impostor_reject_rate = float(impostor_total_reject_windows / impostor_total_windows) if impostor_total_windows else 0.0

    genuine_mean_first_interrupt = (
        float(genuine_first_interrupt_sum / genuine_first_interrupt_n) if genuine_first_interrupt_n else 0.0
    )
    impostor_mean_first_interrupt = (
        float(impostor_first_interrupt_sum / impostor_first_interrupt_n) if impostor_first_interrupt_n else 0.0
    )

    return {
        "vote_window_size": float(window_size),
        "vote_min_rejects": float(min_rejects),
        "threshold": float(threshold),
        "session_far": session_far,
        "session_frr": session_frr,
        "session_tpr": session_tpr,
        "genuine_sessions": float(genuine_sessions),
        "impostor_sessions": float(impostor_sessions),
        "genuine_interrupted_sessions": float(genuine_interrupted_sessions),
        "impostor_detected_sessions": float(impostor_detected_sessions),
        "genuine_total_windows": float(genuine_total_windows),
        "impostor_total_windows": float(impostor_total_windows),
        "genuine_total_reject_windows": float(genuine_total_reject_windows),
        "impostor_total_reject_windows": float(impostor_total_reject_windows),
        "genuine_window_reject_rate": genuine_reject_rate,
        "impostor_window_reject_rate": impostor_reject_rate,
        "genuine_total_interrupts": float(genuine_total_interrupts),
        "impostor_total_interrupts": float(impostor_total_interrupts),
        "genuine_avg_interrupts_per_session": genuine_avg_interrupts,
        "impostor_avg_interrupts_per_session": impostor_avg_interrupts,
        "genuine_mean_first_interrupt_window": genuine_mean_first_interrupt,
        "impostor_mean_first_interrupt_window": impostor_mean_first_interrupt,
    }


def compute_vote_reject_window_metrics_from_arrays(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    window_size: int,
    min_rejects: int,
) -> Dict[str, float]:
    """
    Compute **window-level** metrics for the vote rule, excluding the last (N-1) windows of each session.

    Rule (for an evaluation window start i within a session):
      - reject_window[j] = (score[j] < threshold)
      - predict impostor (0) if, within windows [i, i+N-1], rejects >= M
      - else predict genuine (1)

    We only evaluate start positions i where i+N-1 exists, i.e. i in [0, L-N],
    which matches the requirement to exclude the last (N-1) windows (for N=5, exclude last 4).

    Returns FAR/FRR/ERR as **per-window** rates under the above prediction:
      - FAR: impostor predicted genuine (miss) rate
      - FRR: genuine predicted impostor (false interrupt) rate
      - ERR: overall window error rate

    Conventions:
      - labels: 1=genuine, 0=impostor
      - predicted class: 1=genuine (no lock within next N windows), 0=impostor (lock within next N windows)
    """
    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    window_size = int(window_size)
    min_rejects = int(min_rejects)
    threshold = float(threshold)
    if window_size <= 0 or min_rejects <= 0:
        raise ValueError("window_size/min_rejects must be > 0")
    if min_rejects > window_size:
        raise ValueError("min_rejects must be <= window_size")

    tp = fp = tn = fn = 0
    pos = neg = 0

    genuine_sessions = impostor_sessions = 0
    genuine_first_sum = impostor_first_sum = 0
    genuine_first_n = impostor_first_n = 0

    cur_sid: Optional[int] = None
    cur_label: Optional[int] = None
    cur_rejects: List[int] = []

    def _flush() -> None:
        nonlocal tp, fp, tn, fn, pos, neg
        nonlocal genuine_sessions, impostor_sessions
        nonlocal genuine_first_sum, impostor_first_sum, genuine_first_n, impostor_first_n
        nonlocal cur_sid, cur_label, cur_rejects

        if cur_sid is None or cur_label is None:
            return

        label_i = int(cur_label)
        if label_i == 1:
            genuine_sessions += 1
        else:
            impostor_sessions += 1

        L = len(cur_rejects)
        if L < window_size:
            cur_sid = None
            cur_label = None
            cur_rejects = []
            return

        window_sum = sum(cur_rejects[:window_size])
        first_end_window: Optional[int] = None  # 1-based end-window index

        def _update_counts(pred_genuine: bool) -> None:
            nonlocal tp, fp, tn, fn, pos, neg
            if label_i == 1:
                pos += 1
                if pred_genuine:
                    tp += 1
                else:
                    fn += 1
            else:
                neg += 1
                if pred_genuine:
                    fp += 1
                else:
                    tn += 1

        for i in range(0, L - window_size + 1):
            if i > 0:
                window_sum += cur_rejects[i + window_size - 1] - cur_rejects[i - 1]
            triggered = int(window_sum >= min_rejects)
            pred_genuine = not bool(triggered)
            _update_counts(pred_genuine)
            if first_end_window is None and triggered:
                first_end_window = int(i + window_size)  # end window (1-based)

        if first_end_window is not None:
            if label_i == 1:
                genuine_first_sum += int(first_end_window)
                genuine_first_n += 1
            else:
                impostor_first_sum += int(first_end_window)
                impostor_first_n += 1

        cur_sid = None
        cur_label = None
        cur_rejects = []

    for sid, label, score in zip(session_ids_arr.tolist(), labels_arr.tolist(), scores_arr.tolist()):
        sid_i = int(sid)
        label_i = int(label)
        rejected = 1 if float(score) < threshold else 0

        if cur_sid is None:
            cur_sid = sid_i
            cur_label = label_i
        elif sid_i != int(cur_sid) or (cur_label is not None and label_i != int(cur_label)):
            _flush()
            cur_sid = sid_i
            cur_label = label_i

        cur_rejects.append(int(rejected))

    _flush()

    far = float(fp / max(neg, 1))
    frr = float(fn / max(pos, 1))
    err = float((fp + fn) / max(pos + neg, 1))

    genuine_mean_first = float(genuine_first_sum / genuine_first_n) if genuine_first_n else 0.0
    impostor_mean_first = float(impostor_first_sum / impostor_first_n) if impostor_first_n else 0.0

    return {
        "vote_window_size": float(window_size),
        "vote_min_rejects": float(min_rejects),
        "threshold": float(threshold),
        "far": far,
        "frr": frr,
        "err": err,
        "pos": float(pos),
        "neg": float(neg),
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
        "genuine_sessions": float(genuine_sessions),
        "impostor_sessions": float(impostor_sessions),
        "genuine_first_interrupt_n": float(genuine_first_n),
        "impostor_first_interrupt_n": float(impostor_first_n),
        # 1-based end-window index (trigger happens at the end of that window).
        "genuine_mean_first_interrupt_window": genuine_mean_first,
        "impostor_mean_first_interrupt_window": impostor_mean_first,
    }


def select_threshold_by_vote_window_frr_from_arrays(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    window_size: int,
    min_rejects: int,
    target_window_frr: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Select the *highest* threshold such that window-level FRR <= target_window_frr,
    where FRR is computed by `compute_vote_reject_window_metrics_from_arrays`.
    """
    scores_arr = np.asarray(scores, dtype=np.float64)
    candidates = np.unique(scores_arr)
    if candidates.size == 0:
        raise ValueError("No score candidates found for threshold selection")
    candidates.sort()  # ascending

    target_window_frr = float(target_window_frr)
    if target_window_frr < 0.0:
        target_window_frr = 0.0
    if target_window_frr > 1.0:
        target_window_frr = 1.0

    best_thr = float(candidates[0])
    best_metrics = compute_vote_reject_window_metrics_from_arrays(
        session_ids,
        labels,
        scores,
        threshold=best_thr,
        window_size=int(window_size),
        min_rejects=int(min_rejects),
    )

    lo = 0
    hi = int(candidates.size - 1)
    while lo <= hi:
        mid = (lo + hi) // 2
        thr = float(candidates[mid])
        metrics = compute_vote_reject_window_metrics_from_arrays(
            session_ids,
            labels,
            scores,
            threshold=thr,
            window_size=int(window_size),
            min_rejects=int(min_rejects),
        )
        if float(metrics.get("frr", 0.0)) <= target_window_frr:
            best_thr = thr
            best_metrics = metrics
            lo = mid + 1
        else:
            hi = mid - 1

    return float(best_thr), best_metrics
