from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class FirstInterruptStats:
    sessions: int
    interrupted_sessions: int
    mean_first_interrupt_window: float
    mean_first_interrupt_sec: float
    p_first_interrupt_le_1s: float


def compute_first_interrupt_stats_from_arrays(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    vote_window_size: int,
    vote_min_rejects: int,
    window_size_sec: float,
    overlap: float,
    target_label: int,
    max_decision_time_sec: float = 1.0,
) -> FirstInterruptStats:
    """
    Compute per-session first-interrupt statistics for a given class label.

    Notes:
      - Uses the same vote rule as training evaluation:
          reject = score < threshold
          trigger when rejects_in_last_N >= M
      - Excludes the last (N-1) windows implicitly by only evaluating start
        positions i in [0, L-N].
    """
    first_end_windows = _first_interrupt_end_window_per_session(
        session_ids,
        labels,
        scores,
        threshold=float(threshold),
        vote_window_size=int(vote_window_size),
        vote_min_rejects=int(vote_min_rejects),
        target_label=int(target_label),
    )
    sessions = len(first_end_windows)
    interrupted = [w for w in first_end_windows if w is not None]
    interrupted_sessions = len(interrupted)
    mean_win = float(np.mean([float(w) for w in interrupted])) if interrupted else 0.0
    mean_sec = float(_end_window_to_sec(mean_win, window_size_sec=float(window_size_sec), overlap=float(overlap))) if interrupted else 0.0
    p_le_1s = 0.0
    if sessions:
        threshold_sec = float(max_decision_time_sec)
        le = 0
        for w in interrupted:
            sec = float(_end_window_to_sec(float(w), window_size_sec=float(window_size_sec), overlap=float(overlap)))
            if sec <= threshold_sec:
                le += 1
        p_le_1s = float(le / sessions)

    return FirstInterruptStats(
        sessions=int(sessions),
        interrupted_sessions=int(interrupted_sessions),
        mean_first_interrupt_window=float(mean_win),
        mean_first_interrupt_sec=float(mean_sec),
        p_first_interrupt_le_1s=float(p_le_1s),
    )


def _end_window_to_sec(end_window_1based: float, *, window_size_sec: float, overlap: float) -> float:
    end_window_1based = float(end_window_1based)
    if end_window_1based <= 0.0:
        return 0.0
    stride_sec = float(window_size_sec) * (1.0 - float(overlap))
    return float(window_size_sec + (end_window_1based - 1.0) * stride_sec)


def _first_interrupt_end_window_per_session(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    vote_window_size: int,
    vote_min_rejects: int,
    target_label: int,
) -> List[Optional[int]]:
    """
    Return first interrupt "end-window" index (1-based) for each session of a given label.

    If no interrupt in a session, returns None for that session.
    """
    session_ids_arr = np.asarray(session_ids)
    labels_arr = np.asarray(labels)
    scores_arr = np.asarray(scores)
    if not (session_ids_arr.shape == labels_arr.shape == scores_arr.shape):
        raise ValueError("session_ids/labels/scores must have the same shape")

    vote_window_size = int(vote_window_size)
    vote_min_rejects = int(vote_min_rejects)
    if vote_window_size <= 0 or vote_min_rejects <= 0:
        raise ValueError("vote_window_size/vote_min_rejects must be > 0")
    if vote_min_rejects > vote_window_size:
        raise ValueError("vote_min_rejects must be <= vote_window_size")

    out: List[Optional[int]] = []

    cur_sid: Optional[int] = None
    cur_label: Optional[int] = None
    cur_rejects: List[int] = []

    def _flush() -> None:
        nonlocal cur_sid, cur_label, cur_rejects, out
        if cur_sid is None or cur_label is None:
            return
        if int(cur_label) != int(target_label):
            cur_sid = None
            cur_label = None
            cur_rejects = []
            return

        L = len(cur_rejects)
        if L < vote_window_size:
            out.append(None)
            cur_sid = None
            cur_label = None
            cur_rejects = []
            return

        window_sum = sum(cur_rejects[:vote_window_size])
        first_end: Optional[int] = None
        for i in range(0, L - vote_window_size + 1):
            if i > 0:
                window_sum += cur_rejects[i + vote_window_size - 1] - cur_rejects[i - 1]
            if window_sum >= vote_min_rejects:
                first_end = int(i + vote_window_size)  # 1-based end window index
                break
        out.append(first_end)
        cur_sid = None
        cur_label = None
        cur_rejects = []

    for sid, label, score in zip(session_ids_arr.tolist(), labels_arr.tolist(), scores_arr.tolist()):
        sid_i = int(sid)
        label_i = int(label)
        rejected = 1 if float(score) < float(threshold) else 0

        if cur_sid is None:
            cur_sid = sid_i
            cur_label = label_i
        elif sid_i != int(cur_sid) or (cur_label is not None and label_i != int(cur_label)):
            _flush()
            cur_sid = sid_i
            cur_label = label_i
        cur_rejects.append(int(rejected))

    _flush()
    return out


def first_interrupt_end_windows_per_session(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    vote_window_size: int,
    vote_min_rejects: int,
    target_label: int,
) -> List[Optional[int]]:
    """Public wrapper for `_first_interrupt_end_window_per_session`."""
    return _first_interrupt_end_window_per_session(
        session_ids,
        labels,
        scores,
        threshold=float(threshold),
        vote_window_size=int(vote_window_size),
        vote_min_rejects=int(vote_min_rejects),
        target_label=int(target_label),
    )


def _format_sec_suffix(sec: float) -> str:
    sec_f = float(sec)
    if abs(sec_f - round(sec_f)) < 1e-9:
        return f"{int(round(sec_f))}s"
    text = f"{sec_f:g}"
    text = text.replace(".", "p")
    return f"{text}s"


def first_interrupt_times_sec_per_session(
    session_ids: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    scores: Sequence[float] | np.ndarray,
    *,
    threshold: float,
    vote_window_size: int,
    vote_min_rejects: int,
    window_size_sec: float,
    overlap: float,
    target_label: int,
) -> List[Optional[float]]:
    """
    Return first interrupt time (seconds) per session for a given label.

    - The returned list length equals the number of sessions of `target_label`.
    - Each entry is either the first interrupt time in seconds, or None if the
      session never triggers an interrupt under the vote rule.
    """
    end_windows = first_interrupt_end_windows_per_session(
        session_ids,
        labels,
        scores,
        threshold=float(threshold),
        vote_window_size=int(vote_window_size),
        vote_min_rejects=int(vote_min_rejects),
        target_label=int(target_label),
    )
    out: List[Optional[float]] = []
    for w in end_windows:
        if w is None:
            out.append(None)
            continue
        out.append(float(_end_window_to_sec(float(w), window_size_sec=float(window_size_sec), overlap=float(overlap))))
    return out


def probability_first_interrupt_within(
    first_interrupt_times_sec: Iterable[Optional[float]],
    *,
    threshold_sec: float,
) -> float:
    times = list(first_interrupt_times_sec)
    if not times:
        return 0.0
    thr = float(threshold_sec)
    hit = 0
    for t in times:
        if t is not None and float(t) <= thr:
            hit += 1
    return float(hit / len(times))


def p_first_interrupt_le_columns(
    prefix: str,
    first_interrupt_times_sec: Iterable[Optional[float]],
    thresholds_sec: Sequence[float],
) -> Dict[str, float]:
    """
    Build columns like:
      {\"<prefix>_p_first_interrupt_le_2s\": 0.5, ...}
    """
    out: Dict[str, float] = {}
    for sec in thresholds_sec:
        key = f"{prefix}_p_first_interrupt_le_{_format_sec_suffix(float(sec))}"
        out[key] = probability_first_interrupt_within(first_interrupt_times_sec, threshold_sec=float(sec))
    return out


def as_dict(prefix: str, stats: FirstInterruptStats) -> Dict[str, float]:
    return {
        f"{prefix}_sessions": float(stats.sessions),
        f"{prefix}_interrupted_sessions": float(stats.interrupted_sessions),
        f"{prefix}_mean_first_interrupt_window": float(stats.mean_first_interrupt_window),
        f"{prefix}_mean_first_interrupt_sec": float(stats.mean_first_interrupt_sec),
        f"{prefix}_p_first_interrupt_le_1s": float(stats.p_first_interrupt_le_1s),
    }
