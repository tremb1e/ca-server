import math
from pathlib import Path

import pytest

from src.ca_config import load_ca_config
from src.utils.secondary_hysteresis import SecondaryHysteresisConfig, SecondaryHysteresisTracker


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def _double_ema(scores: list[float], alpha: float) -> float:
    """Mirror the tracker's score-domain EMA recurrence exactly."""
    ema: float | None = None
    for score in scores:
        ema = score if ema is None else alpha * score + (1.0 - alpha) * ema
    assert ema is not None
    return ema


def test_secondary_vote_emits_after_ten_inputs_and_rejects_at_eight() -> None:
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="vote",
            vote_window_size=10,
            vote_min_rejects=8,
        )
    )

    primary_threshold = 0.45
    # Eight rejects (score < threshold) followed by two accepts (score >= threshold).
    primary_scores = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.70, 0.72]
    results = [
        tracker.update(
            accepted=score >= primary_threshold,
            interrupt=False,
            primary_score=score,
            primary_threshold=primary_threshold,
        )
        for score in primary_scores
    ]

    assert all(result is None for result in results[:-1])
    final = results[-1]
    assert final is not None
    assert final.strategy == "vote"
    assert final.reject_count == 8
    assert final.accept is False
    assert final.interrupt is True
    # Score is the 8th smallest (M-th order statistic) window score, on the policy axis.
    assert final.score == pytest.approx(sorted(primary_scores)[8 - 1])
    assert final.threshold == pytest.approx(primary_threshold)
    # Verdict <-> score self-consistency: reject_count >= min_rejects <=> boundary < threshold.
    assert (final.score >= final.threshold) == final.accept


def test_secondary_vote_accepts_when_rejects_below_minimum() -> None:
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="vote",
            vote_window_size=10,
            vote_min_rejects=8,
        )
    )

    primary_threshold = 0.45
    # Seven rejects, three accepts -> reject_count (7) < min_rejects (8) -> accept.
    primary_scores = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.70, 0.72, 0.74]
    final = None
    for score in primary_scores:
        final = tracker.update(
            accepted=score >= primary_threshold,
            interrupt=False,
            primary_score=score,
            primary_threshold=primary_threshold,
        )

    assert final is not None
    assert final.reject_count == 7
    assert final.accept is True
    assert final.interrupt is False
    # 8th smallest score is now an accept score (>= threshold).
    assert final.score == pytest.approx(sorted(primary_scores)[8 - 1])
    assert final.threshold == pytest.approx(primary_threshold)
    assert (final.score >= final.threshold) == final.accept


def test_secondary_vote_score_is_mth_order_statistic_on_policy_axis() -> None:
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="vote",
            vote_window_size=5,
            vote_min_rejects=2,
        )
    )

    primary_threshold = 0.45
    primary_scores = [0.52, 0.58, 0.41, 0.60, 0.55]
    final = None
    for score in primary_scores:
        final = tracker.update(
            accepted=score >= primary_threshold,
            interrupt=False,
            primary_score=score,
            primary_threshold=primary_threshold,
        )

    assert final is not None
    assert final.strategy == "vote"
    assert final.input_count == 5
    assert final.reject_count == 1  # only 0.41 < 0.45
    # M-th smallest (2nd) score reported on the policy axis.
    assert final.score == pytest.approx(sorted(primary_scores)[2 - 1])  # 0.52
    assert final.threshold == pytest.approx(primary_threshold)
    assert final.accept is True
    assert (final.score >= final.threshold) == final.accept


def test_secondary_ema_emits_by_configured_decision_time() -> None:
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="ema",
            primary_result_interval_sec=1.0,
            decision_time_sec=3.0,
            ema_alpha=1.0,
            ema_reject_threshold=0.8,
        )
    )

    primary_threshold = 0.5
    accept_kwargs = dict(primary_score=1.0, primary_threshold=primary_threshold)
    assert tracker.update(accepted=True, interrupt=False, **accept_kwargs) is None
    assert tracker.update(accepted=True, interrupt=False, **accept_kwargs) is None
    accepted = tracker.update(accepted=True, interrupt=False, **accept_kwargs)
    assert accepted is not None
    assert accepted.accept is True
    # alpha=1.0 -> ema score tracks the latest primary score on the policy axis.
    assert accepted.score == pytest.approx(1.0)
    assert accepted.threshold == pytest.approx(primary_threshold)
    assert (accepted.score >= accepted.threshold) == accepted.accept

    reject_kwargs = dict(primary_score=0.0, primary_threshold=primary_threshold)
    assert tracker.update(accepted=False, interrupt=False, **reject_kwargs) is None
    assert tracker.update(accepted=False, interrupt=False, **reject_kwargs) is None
    rejected = tracker.update(accepted=False, interrupt=True, **reject_kwargs)
    assert rejected is not None
    assert rejected.accept is False
    assert rejected.interrupt is True
    assert rejected.score == pytest.approx(0.0)
    assert rejected.threshold == pytest.approx(primary_threshold)
    assert rejected.reject_count == 3
    assert (rejected.score >= rejected.threshold) == rejected.accept


def test_secondary_ema_score_equals_double_ema_of_primary_score() -> None:
    alpha = 0.5
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="ema",
            primary_result_interval_sec=1.0,
            decision_time_sec=4.0,
            ema_alpha=alpha,
            ema_reject_threshold=0.8,
        )
    )

    primary_threshold = 0.45
    primary_scores = [0.60, 0.58, 0.30, 0.62]
    final = None
    for score in primary_scores:
        final = tracker.update(
            accepted=score >= primary_threshold,
            interrupt=False,
            primary_score=score,
            primary_threshold=primary_threshold,
        )

    assert final is not None
    assert final.strategy == "ema"
    expected_score = _double_ema(primary_scores, alpha)
    assert final.score == pytest.approx(expected_score)
    assert final.threshold == pytest.approx(primary_threshold)
    assert final.accept == (final.score >= final.threshold)
    # reject EMA is telemetry only and is distinct from the reported decision score.
    assert final.ema_reject_score is not None
    assert final.ema_reject_score != pytest.approx(final.score)


def test_secondary_legit_user_score_stays_on_policy_axis_not_reject_ratio() -> None:
    """Scale-fix regression: legit primary scores above threshold must accept and
    report a real (0,1) policy-axis score, NOT a fixed reject-ratio (0.6/0.3/0.2)."""
    window_size = 10
    min_rejects = 8
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="vote",
            vote_window_size=window_size,
            vote_min_rejects=min_rejects,
        )
    )

    primary_threshold = 0.45
    # Legitimate user: every window score is comfortably above the policy threshold.
    primary_scores = [0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.51, 0.53, 0.55, 0.57]
    final = None
    for score in primary_scores:
        assert score >= primary_threshold
        final = tracker.update(
            accepted=True,
            interrupt=False,
            primary_score=score,
            primary_threshold=primary_threshold,
        )

    assert final is not None
    assert final.accept is True
    assert final.interrupt is False
    assert final.reject_count == 0
    # Reported score is on the policy axis (>= threshold) and self-consistent.
    assert final.threshold == pytest.approx(primary_threshold)
    assert final.score >= final.threshold
    assert (final.score >= final.threshold) == final.accept
    assert final.score == pytest.approx(sorted(primary_scores)[min_rejects - 1])  # 0.57
    # The score must NOT be a reject-ratio quantity from the old scale.
    assert final.score != pytest.approx((window_size - min_rejects + 1) / window_size)  # 0.3
    assert final.score != pytest.approx(1.0 - 0.8)  # 0.2
    # And it lives strictly inside the open (0, 1) score interval.
    assert 0.0 < final.score < 1.0


def test_secondary_legit_user_score_stays_on_policy_axis_ema() -> None:
    alpha = 0.5
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="ema",
            primary_result_interval_sec=1.0,
            decision_time_sec=4.0,
            ema_alpha=alpha,
            ema_reject_threshold=0.8,
        )
    )

    primary_threshold = 0.45
    primary_scores = [0.50, 0.55, 0.58, 0.60]
    final = None
    for score in primary_scores:
        final = tracker.update(
            accepted=True,
            interrupt=False,
            primary_score=score,
            primary_threshold=primary_threshold,
        )

    assert final is not None
    assert final.accept is True
    assert final.threshold == pytest.approx(primary_threshold)
    assert final.score == pytest.approx(_double_ema(primary_scores, alpha))
    assert final.score >= final.threshold
    assert final.score != pytest.approx(1.0 - 0.8)  # not the old ema reject-ratio scale
    assert 0.0 < final.score < 1.0


def test_secondary_attacker_score_below_threshold_rejects() -> None:
    primary_threshold = 0.45

    # Vote: attacker scores far below threshold -> all windows reject.
    vote_tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="vote",
            vote_window_size=5,
            vote_min_rejects=2,
        )
    )
    attacker_scores = [0.10, 0.08, 0.12, 0.05, 0.09]
    vote_final = None
    for score in attacker_scores:
        vote_final = vote_tracker.update(
            accepted=False,
            interrupt=True,
            primary_score=score,
            primary_threshold=primary_threshold,
        )
    assert vote_final is not None
    assert vote_final.accept is False
    assert vote_final.reject_count == 5
    assert vote_final.score < vote_final.threshold
    assert (vote_final.score >= vote_final.threshold) == vote_final.accept

    # EMA: attacker score EMA stays below threshold -> reject.
    ema_tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="ema",
            primary_result_interval_sec=1.0,
            decision_time_sec=4.0,
            ema_alpha=0.5,
            ema_reject_threshold=0.8,
        )
    )
    ema_final = None
    for score in [0.10, 0.12, 0.08, 0.05]:
        ema_final = ema_tracker.update(
            accepted=False,
            interrupt=True,
            primary_score=score,
            primary_threshold=primary_threshold,
        )
    assert ema_final is not None
    assert ema_final.accept is False
    assert ema_final.score < ema_final.threshold
    assert (ema_final.score >= ema_final.threshold) == ema_final.accept


def test_secondary_score_verdict_self_consistency_property() -> None:
    import random

    rng = random.Random(20240627)
    primary_threshold = 0.45

    # EMA strategy: score-domain verdict is strictly score >= threshold.
    for _ in range(80):
        tracker = SecondaryHysteresisTracker(
            SecondaryHysteresisConfig(
                strategy="ema",
                primary_result_interval_sec=1.0,
                decision_time_sec=rng.choice([2.0, 3.0, 4.0, 5.0]),
                ema_alpha=rng.choice([0.2, 0.3, 0.5, 0.8, 1.0]),
                ema_reject_threshold=0.8,
            )
        )
        emitted = 0
        for _ in range(rng.randint(8, 40)):
            score = rng.uniform(0.0, 1.0)
            result = tracker.update(
                accepted=score >= primary_threshold,
                interrupt=False,
                primary_score=score,
                primary_threshold=primary_threshold,
            )
            if result is not None:
                emitted += 1
                assert (result.score >= result.threshold) == result.accept
        assert emitted > 0

    # Vote strategy under the standard feeding rejected == (primary_score < primary_threshold):
    # the count-based verdict equals the M-th order statistic vs threshold comparison.
    for _ in range(80):
        window_size = rng.randint(2, 12)
        min_rejects = rng.randint(1, window_size)
        tracker = SecondaryHysteresisTracker(
            SecondaryHysteresisConfig(
                strategy="vote",
                vote_window_size=window_size,
                vote_min_rejects=min_rejects,
            )
        )
        emitted = 0
        for _ in range(window_size * rng.randint(1, 4)):
            score = rng.uniform(0.0, 1.0)
            result = tracker.update(
                accepted=score >= primary_threshold,
                interrupt=False,
                primary_score=score,
                primary_threshold=primary_threshold,
            )
            if result is not None:
                emitted += 1
                assert (result.score >= result.threshold) == result.accept
        assert emitted > 0


def test_secondary_config_reports_effective_vote_decision_time() -> None:
    cfg = SecondaryHysteresisConfig(
        strategy="vote",
        primary_result_interval_sec=1.5,
        decision_time_sec=10.0,
        vote_window_size=4,
        vote_min_rejects=3,
    )

    assert cfg.vote_effective_window_size == 4
    assert cfg.effective_decision_time_sec == 6.0


def test_secondary_ema_reject_count_is_actual_period_count() -> None:
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="ema",
            primary_result_interval_sec=1.0,
            decision_time_sec=3.0,
            ema_alpha=0.5,
            ema_reject_threshold=0.8,
        )
    )

    primary_threshold = 0.45
    assert (
        tracker.update(accepted=False, interrupt=True, primary_score=0.10, primary_threshold=primary_threshold)
        is None
    )
    assert (
        tracker.update(accepted=True, interrupt=False, primary_score=0.80, primary_threshold=primary_threshold)
        is None
    )
    result = tracker.update(
        accepted=False, interrupt=True, primary_score=0.12, primary_threshold=primary_threshold
    )

    assert result is not None
    # reject_count is telemetry: actual rejects observed in the emit period (2 of 3).
    assert result.reject_count == 2


def test_ca_config_loads_secondary_hysteresis_fields(tmp_path: Path) -> None:
    cfg_path = tmp_path / "ca_config.toml"
    cfg_path.write_text(
        """
[auth]
secondary_hysteresis_enabled = true
secondary_hysteresis_strategy = "ema"
primary_result_interval_sec = 1.0
secondary_decision_time_sec = 12.0
secondary_vote_window_size = 12
secondary_vote_min_rejects = 9
secondary_ema_alpha = 0.4
secondary_ema_reject_threshold = 0.7
""",
        encoding="utf-8",
    )

    cfg = load_ca_config(cfg_path)

    assert cfg.auth.secondary_hysteresis_enabled is True
    assert cfg.auth.secondary_hysteresis_strategy == "ema"
    assert cfg.auth.primary_result_interval_sec == 1.0
    assert cfg.auth.secondary_decision_time_sec == 12.0
    assert cfg.auth.secondary_vote_window_size == 12
    assert cfg.auth.secondary_vote_min_rejects == 9
    assert cfg.auth.secondary_ema_alpha == 0.4
    assert cfg.auth.secondary_ema_reject_threshold == 0.7
