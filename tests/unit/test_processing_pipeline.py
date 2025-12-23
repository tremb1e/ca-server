import os
import time
from pathlib import Path

import pytest

from src.processing.pipeline import (
    ProcessingConfig,
    _list_sessions_for_user,
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
        workers=1,
        window_workers=1,
        hmog_val_subject_count=0,
        hmog_test_subject_count=0,
        hmog_max_rows_per_subject=None,
        hmog_max_rows_total=None,
    )


def _write_dummy_file(path: Path, size_bytes: int) -> None:
    path.write_bytes(b"0" * size_bytes)


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
            "mag_x": [0.0] * rows,
            "mag_y": [0.0] * rows,
            "mag_z": [0.0] * rows,
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
