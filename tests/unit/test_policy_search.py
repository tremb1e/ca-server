import numpy as np

from src.policy_search.metrics import compute_first_interrupt_stats_from_arrays
from src.policy_search.pareto import ParetoPoint, pareto_frontier


def test_pareto_frontier_minimization() -> None:
    pts = [
        ParetoPoint((1.0, 1.0), {"id": "a"}),
        ParetoPoint((0.9, 1.1), {"id": "b"}),  # trade-off
        ParetoPoint((0.8, 0.8), {"id": "c"}),  # dominates a and b
        ParetoPoint((0.8, 0.9), {"id": "d"}),  # dominated by c
    ]
    front = pareto_frontier(pts)
    assert [p.payload["id"] for p in front] == ["c"]


def test_first_interrupt_stats_vote_rule() -> None:
    # Two impostor sessions (label=0), each with 10 windows.
    # vote: N=10, M=6 => session0 triggers at end-window=10, session1 never triggers.
    session_ids = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    labels = np.array([0] * 20, dtype=np.int8)
    # threshold=0.0; score<0 => reject
    scores = np.array(
        # session 0: 6 rejects across the only 10-window span
        [-1, -1, -1, -1, -1, -1, 1, 1, 1, 1]
        # session 1: no rejects
        + [1] * 10,
        dtype=np.float32,
    )

    stats = compute_first_interrupt_stats_from_arrays(
        session_ids,
        labels,
        scores,
        threshold=0.0,
        vote_window_size=10,
        vote_min_rejects=6,
        window_size_sec=0.1,
        overlap=0.5,
        target_label=0,
        max_decision_time_sec=1.0,
    )
    assert stats.sessions == 2
    assert stats.interrupted_sessions == 1
    assert stats.mean_first_interrupt_window == 10.0
    assert abs(stats.mean_first_interrupt_sec - 0.55) < 1e-9
    assert abs(stats.p_first_interrupt_le_1s - 0.5) < 1e-9


def test_first_interrupt_stats_excludes_last_n_minus_1_windows() -> None:
    # Session shorter than N should not be evaluated => no interrupt.
    session_ids = np.array([0] * 9, dtype=np.int32)
    labels = np.array([0] * 9, dtype=np.int8)
    scores = np.array([-1] * 9, dtype=np.float32)

    stats = compute_first_interrupt_stats_from_arrays(
        session_ids,
        labels,
        scores,
        threshold=0.0,
        vote_window_size=10,
        vote_min_rejects=6,
        window_size_sec=0.1,
        overlap=0.5,
        target_label=0,
        max_decision_time_sec=1.0,
    )
    assert stats.sessions == 1
    assert stats.interrupted_sessions == 0
    assert stats.mean_first_interrupt_window == 0.0
    assert stats.mean_first_interrupt_sec == 0.0
    assert stats.p_first_interrupt_le_1s == 0.0


def test_prompt_vote_min_rejects_ranges_match_spec() -> None:
    from src.policy_search.runner import PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N

    assert PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N[7] == (4, 6)
    assert PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N[10] == (6, 8)
    assert PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N[16] == (9, 13)
    assert PROMPT_VOTE_MIN_REJECTS_RANGE_BY_N[20] == (11, 16)
