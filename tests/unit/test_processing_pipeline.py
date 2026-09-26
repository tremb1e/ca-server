import os
import time
from pathlib import Path

import pytest

from src.processing.pipeline import (
    ProcessingConfig,
    _ensure_split_has_both_classes,
    _load_hmog_attackers,
    _list_sessions_for_user,
    _resolve_hmog_root,
    _select_hmog_attacker_ids,
    _select_sessions_for_processing,
    _write_windows,
)


def _make_cfg(tmp_path: Path, *, min_total_bytes: int, target_total_bytes: int) -> ProcessingConfig:
    raw_root = tmp_path / "raw"
    processed_root = tmp_path / "processed"
    hmog_root = tmp_path / "hmog"
    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    hmog_root.mkdir(parents=True, exist_ok=True)
    return ProcessingConfig(
        raw_root=raw_root,
        processed_root=processed_root,
        zscore_root=processed_root / "z-score",
        window_root=processed_root / "window",
        hmog_root=hmog_root,
        sampling_rate_hz=100,
        window_sizes=[0.1],
        window_overlap=0.5,
        min_total_bytes=min_total_bytes,
        target_total_bytes=target_total_bytes,
        train_ratio=0.75,
        val_ratio=0.125,
        test_ratio=0.125,
        workers=1,
        window_workers=1,
        process_user_workers=1,
        hmog_val_subject_count=0,
        hmog_test_subject_count=0,
        hmog_min_subject_count=10,
        hmog_session_count=24,
        hmog_balance_ratio=1.0,
        hmog_max_rows_per_subject=None,
        hmog_max_rows_total=None,
    )


def _write_dummy_file(path: Path, size_bytes: int) -> None:
    path.write_bytes(b"0" * size_bytes)


def _sensor_records(timestamps):
    return {
        sensor: [{"timestamp": timestamp, "x": 1.0, "y": 2.0, "z": 3.0} for timestamp in timestamps]
        for sensor in ("acc", "gyr", "mag")
    }


def test_resampling_does_not_create_samples_during_long_outages(tmp_path):
    from src.processing.pipeline import _resample_records

    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    timestamps = list(range(1000, 1300, 10)) + list(range(86_401_000, 86_401_300, 10))
    frame = _resample_records(_sensor_records(timestamps), "session", "u", cfg)
    assert frame["timestamp"].tolist() == timestamps
    assert frame["session"].nunique() == 2
    output = tmp_path / "windows.csv"
    assert _write_windows(frame, output, 0.2, 100, 0.5) == 80  # two windows per run


def test_resampling_drops_missing_motion_but_preserves_magnetic_gaps(tmp_path):
    from src.processing.pipeline import _resample_records

    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    records = _sensor_records(range(1000, 2000, 10))
    records["gyr"] = [row for row in records["gyr"] if not 1300 <= row["timestamp"] < 1600]
    records["mag"] = [row for row in records["mag"] if not 1700 <= row["timestamp"] < 1990]
    frame = _resample_records(records, "session", "u", cfg)
    assert len(frame) == 70
    assert not frame["timestamp"].between(1300, 1599).any()
    assert frame.loc[frame["timestamp"].between(1700, 1989), "mag_x"].isna().all()


def test_resampling_retains_short_interpolation_and_uses_common_coverage(tmp_path):
    from src.processing.pipeline import _resample_records

    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    records = _sensor_records([1001, 1021, 1041])
    records["gyr"] = records["gyr"][1:]
    frame = _resample_records(records, "session", "u", cfg)
    assert frame["timestamp"].tolist() == [1020, 1030, 1040]
    assert frame["acc_x"].tolist() == [1.0, 1.0, 1.0]


def test_clock_reset_inside_packet_preserves_capture_order(tmp_path):
    import json
    from src.processing.pipeline import _assemble_user_dataframe

    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    user_dir = cfg.raw_root / "u"
    user_dir.mkdir()
    samples = []
    for start, value in [(10_000, 1.0), (1000, 2.0)]:
        for timestamp in range(start, start + 300, 10):
            for sensor in (1, 2, 3):
                samples.append({"sensor_type": sensor, "timestamp_ns": timestamp * 1_000_000,
                                "x": value, "y": value, "z": value})
    path = user_dir / "session.jsonl"
    path.write_text(json.dumps({"sensor_data": samples}) + "\n")
    frame = _assemble_user_dataframe("u", [path], cfg)
    assert len(frame) == 60
    assert frame["acc_x"].tolist() == [1.0] * 30 + [2.0] * 30
    assert frame["session"].nunique() == 2
    assert "_source_session" not in frame


def test_retransmitted_samples_do_not_create_clock_domains():
    from src.processing.pipeline import _clock_record_groups

    def packet(timestamps):
        return {"sensor_data": [
            {"sensor_type": sensor, "timestamp_ns": timestamp * 1_000_000,
             "x": 1, "y": 2, "z": 3}
            for timestamp in timestamps for sensor in (1, 2, 3)
        ]}

    groups = list(_clock_record_groups([packet([1000, 1010]), packet([1000, 1010, 1020])]))
    assert len(groups) == 1
    assert [row["timestamp"] for row in groups[0]["acc"]] == [1000, 1010, 1020]


def test_reboot_metadata_preserves_repeated_timestamps_in_new_clock():
    from src.processing.pipeline import _clock_record_groups

    def packet(wall_ms, uptime_ns, value):
        return {
            "base_wall_ms": wall_ms, "device_uptime_ns": uptime_ns,
            "sensor_data": [
                {"sensor_type": sensor, "timestamp_ns": timestamp * 1_000_000,
                 "x": value, "y": value, "z": value}
                for timestamp in (1000, 1010) for sensor in (1, 2, 3)
            ],
        }

    first = packet(10000, 2_000_000_000, 1)
    # Same samples retried later in the same boot remain duplicates.
    retry = packet(10100, 2_100_000_000, 1)
    reboot = packet(20000, 1_100_000_000, 2)
    groups = list(_clock_record_groups([first, retry, reboot]))
    assert len(groups) == 2
    assert [row["x"] for row in groups[0]["acc"]] == [1, 1]
    assert [row["x"] for row in groups[1]["acc"]] == [2, 2]


def test_windowing_does_not_bridge_gaps_in_existing_csv(tmp_path):
    import pandas as pd
    from src.processing.pipeline import _resample_records

    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    frame = _resample_records(_sensor_records(range(1000, 1200, 10)), "s", "u", cfg)
    second = frame.copy()
    second["timestamp"] += 10_000
    frame = pd.concat([frame, second], ignore_index=True)
    output = tmp_path / "windows.csv"
    assert _write_windows(frame, output, 0.2, 100, 0.5) == 40
    windows = pd.read_csv(output)
    assert windows.groupby("window_id")["timestamp"].agg(lambda x: x.max() - x.min()).tolist() == [190, 190]


def test_select_sessions_uses_earliest_prefix_to_hit_target(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=100)
    user_dir = cfg.raw_root / "user1"
    user_dir.mkdir(parents=True, exist_ok=True)

    files = [
        user_dir / "session_1700000000000.jsonl",
        user_dir / "session_1700000000100.jsonl",
        user_dir / "session_1700000000200.jsonl",
        user_dir / "session_1700000000300.jsonl",
    ]
    sizes = [10, 60, 40, 20]
    for path, size in zip(files, sizes, strict=True):
        _write_dummy_file(path, size)

    ordered = _list_sessions_for_user(user_dir)
    assert [p.name for p in ordered] == [p.name for p in files]

    selected = _select_sessions_for_processing(ordered, cfg)
    assert [p.name for p in selected] == [p.name for p in files[:3]]


def test_select_sessions_skips_when_below_threshold(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, min_total_bytes=100, target_total_bytes=50)
    user_dir = cfg.raw_root / "user1"
    user_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        user_dir / "session_1700000000000.jsonl",
        user_dir / "session_1700000000100.jsonl",
    ]
    for path in paths:
        _write_dummy_file(path, 20)  # total 40 < min_total_bytes=100

    ordered = _list_sessions_for_user(user_dir)
    assert _select_sessions_for_processing(ordered, cfg) == []


def test_select_sessions_includes_earlier_sessions_even_if_next_is_large(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=100)
    user_dir = cfg.raw_root / "user1"
    user_dir.mkdir(parents=True, exist_ok=True)

    small = user_dir / "session_1700000000000.jsonl"
    large = user_dir / "session_1700000000100.jsonl"
    _write_dummy_file(small, 10)
    _write_dummy_file(large, 150)

    ordered = _list_sessions_for_user(user_dir)
    selected = _select_sessions_for_processing(ordered, cfg)
    assert [p.name for p in selected] == [small.name, large.name]


def test_list_sessions_prefers_filename_timestamp_over_mtime(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    user_dir = cfg.raw_root / "user1"
    user_dir.mkdir(parents=True, exist_ok=True)

    older = user_dir / "session_1700000000000.jsonl"
    newer = user_dir / "session_1700000000100.jsonl"
    _write_dummy_file(older, 1)
    _write_dummy_file(newer, 1)

    now = time.time()
    os.utime(older, (now, now))  # older timestamp, but newer mtime
    os.utime(newer, (now, now - 3600))  # newer timestamp, but older mtime

    ordered = _list_sessions_for_user(user_dir)
    assert [p.name for p in ordered] == [older.name, newer.name]


def test_write_windows_generates_fixed_length_windows_without_crossing_sessions(tmp_path: Path) -> None:
    import pandas as pd

    window_size = 0.1
    sampling_rate = 100
    window_points = int(window_size * sampling_rate)

    def make_session(session: str, start_ts: int) -> pd.DataFrame:
        rows = 25
        timestamps = [start_ts + i * 10 for i in range(rows)]
        base = {
            "subject": ["u1"] * rows,
            "session": [session] * rows,
            "timestamp": timestamps,
            "acc_x": [0.0] * rows,
            "acc_y": [0.0] * rows,
            "acc_z": [0.0] * rows,
            "gyr_x": [0.0] * rows,
            "gyr_y": [0.0] * rows,
            "gyr_z": [0.0] * rows,
        }
        return pd.DataFrame(base)

    df = pd.concat(
        [
            make_session("s1", 0),
            make_session("s2", 10_000),
        ],
        ignore_index=True,
    )

    out_path = tmp_path / "windows.csv"
    rows_written = _write_windows(df, out_path, window_size, sampling_rate, overlap=0.5)
    assert rows_written > 0

    out_df = pd.read_csv(out_path)
    counts = out_df.groupby("window_id").size()
    assert counts.min() == window_points
    assert counts.max() == window_points
    assert sorted(counts.index.tolist()) == list(range(len(counts)))
    assert out_df.groupby("window_id")["session"].nunique().max() == 1


def test_resolve_hmog_root_uses_local_fallback(tmp_path: Path) -> None:
    processed_root = tmp_path / "processed_data"
    processed_root.mkdir(parents=True, exist_ok=True)
    fallback_root = tmp_path / "hmog_preprocessed"
    fallback_root.mkdir(parents=True, exist_ok=True)

    resolved = _resolve_hmog_root(tmp_path / "missing_hmog", processed_root)
    assert resolved == fallback_root


def test_select_hmog_attacker_ids_uses_fixed_rank_ranges() -> None:
    subjects = [str(i) for i in range(1, 31)]
    val_ids, test_ids = _select_hmog_attacker_ids(subjects)
    assert val_ids == [str(i) for i in range(1, 11)]
    assert test_ids == [str(i) for i in range(21, 31)]


def test_load_hmog_attackers_keeps_only_requested_session_range(tmp_path: Path) -> None:
    import pandas as pd

    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    for sid in ("100669", "151985"):
        subject_dir = cfg.hmog_root / sid
        subject_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            {
                "subject": [sid, sid, sid],
                "session": [f"{sid}_session_1", f"{sid}_session_6", f"{sid}_session_7"],
                "timestamp": [1, 2, 3],
                "acc_x": [0.1, 0.2, 0.3],
                "acc_y": [0.1, 0.2, 0.3],
                "acc_z": [0.1, 0.2, 0.3],
                "gyr_x": [0.1, 0.2, 0.3],
                "gyr_y": [0.1, 0.2, 0.3],
                "gyr_z": [0.1, 0.2, 0.3],
                "mag_x": [30.0, 31.0, 32.0],
                "mag_y": [10.0, 11.0, 12.0],
                "mag_z": [20.0, 21.0, 22.0],
            }
        )
        df.to_csv(subject_dir / f"{sid}_train.csv", index=False)

    attackers = _load_hmog_attackers(["100669", "151985"], cfg, session_range=(1, 6))
    assert not attackers.empty
    assert set(attackers["subject"].astype(str).unique().tolist()) == {"100669", "151985"}
    sessions = attackers["session"].astype(str).tolist()
    assert all(s.endswith("_session_1") or s.endswith("_session_6") for s in sessions)
    assert not any(s.endswith("_session_7") for s in sessions)


def test_load_hmog_attackers_balances_to_target_rows(tmp_path: Path) -> None:
    import pandas as pd

    cfg = _make_cfg(tmp_path, min_total_bytes=0, target_total_bytes=0)
    for sid in ("100669", "151985"):
        subject_dir = cfg.hmog_root / sid
        subject_dir.mkdir(parents=True, exist_ok=True)
        rows = 20
        df = pd.DataFrame(
            {
                "subject": [sid] * rows,
                "session": [f"{sid}_session_1"] * rows,
                "timestamp": list(range(rows)),
                "acc_x": [0.1] * rows,
                "acc_y": [0.1] * rows,
                "acc_z": [0.1] * rows,
                "gyr_x": [0.1] * rows,
                "gyr_y": [0.1] * rows,
                "gyr_z": [0.1] * rows,
                "mag_x": [30.0] * rows,
                "mag_y": [10.0] * rows,
                "mag_z": [20.0] * rows,
            }
        )
        df.to_csv(subject_dir / f"{sid}_train.csv", index=False)

    attackers = _load_hmog_attackers(["100669", "151985"], cfg, session_range=(1, 24), target_rows=12)
    assert len(attackers) == 12
    assert attackers.groupby("subject").size().to_dict() == {"100669": 6, "151985": 6}


def test_ensure_split_has_both_classes_rejects_single_class() -> None:
    import pandas as pd

    df = pd.DataFrame({"subject": ["u1", "u1"], "session": ["s1", "s1"], "timestamp": [1, 2]})
    with pytest.raises(ValueError, match="must contain both classes"):
        _ensure_split_has_both_classes(df, split_name="val", user_id="u1", hmog_root=Path("/tmp/hmog"))


def test_ensure_split_has_both_classes_accepts_mixed_subjects() -> None:
    import pandas as pd

    df = pd.DataFrame({"subject": ["u1", "attacker"], "session": ["s1", "s2"], "timestamp": [1, 2]})
    _ensure_split_has_both_classes(df, split_name="test", user_id="u1", hmog_root=Path("/tmp/hmog"))
