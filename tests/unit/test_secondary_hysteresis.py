from pathlib import Path

from src.ca_config import load_ca_config
from src.utils.secondary_hysteresis import SecondaryHysteresisConfig, SecondaryHysteresisTracker


def test_secondary_vote_emits_after_ten_inputs_and_rejects_at_eight() -> None:
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="vote",
            vote_window_size=10,
            vote_min_rejects=8,
        )
    )

    results = [
        tracker.update(accepted=i >= 8, interrupt=False)
        for i in range(10)
    ]

    assert all(result is None for result in results[:-1])
    final = results[-1]
    assert final is not None
    assert final.strategy == "vote"
    assert final.reject_count == 8
    assert final.accept is False
    assert final.interrupt is True
    assert final.score == 0.2
    assert final.threshold == 0.3


def test_secondary_vote_accepts_when_rejects_below_minimum() -> None:
    tracker = SecondaryHysteresisTracker(
        SecondaryHysteresisConfig(
            strategy="vote",
            vote_window_size=10,
            vote_min_rejects=8,
        )
    )

    final = None
    for i in range(10):
        final = tracker.update(accepted=i >= 7, interrupt=False)

    assert final is not None
    assert final.reject_count == 7
    assert final.accept is True
    assert final.interrupt is False


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

    assert tracker.update(accepted=True, interrupt=False) is None
    assert tracker.update(accepted=True, interrupt=False) is None
    accepted = tracker.update(accepted=True, interrupt=False)
    assert accepted is not None
    assert accepted.accept is True
    assert accepted.score == 1.0

    assert tracker.update(accepted=False, interrupt=False) is None
    assert tracker.update(accepted=False, interrupt=False) is None
    rejected = tracker.update(accepted=False, interrupt=True)
    assert rejected is not None
    assert rejected.accept is False
    assert rejected.interrupt is True
    assert rejected.score == 0.0
    assert rejected.reject_count == 3


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

    assert tracker.update(accepted=False, interrupt=True) is None
    assert tracker.update(accepted=True, interrupt=False) is None
    result = tracker.update(accepted=False, interrupt=True)

    assert result is not None
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
