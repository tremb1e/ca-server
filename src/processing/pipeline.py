import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..ca_config import get_ca_config
from ..config import settings
from .scaler import apply_scaler, write_scaler

logger = logging.getLogger(__name__)


SENSOR_TYPE_TO_PREFIX = {
    1: "acc",
    2: "gyr",
    3: "mag",
    "accelerometer": "acc",
    "gyroscope": "gyr",
    "magnetometer": "mag",
}


FEATURE_COLUMNS = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "mag_x",
    "mag_y",
    "mag_z",
]


# Windowing can generate very large intermediate outputs. Flush to disk in
# batches to avoid unbounded memory growth.
WINDOW_BATCH_ROWS = 500_000


@dataclass(frozen=True)
class ProcessingConfig:
    raw_root: Path
    processed_root: Path
    zscore_root: Path
    window_root: Path
    hmog_root: Path
    sampling_rate_hz: int
    window_sizes: Sequence[float]
    window_overlap: float
    min_total_bytes: int
    target_total_bytes: int
    workers: int
    window_workers: int
    hmog_val_subject_count: int
    hmog_test_subject_count: int
    hmog_max_rows_per_subject: Optional[int]
    hmog_max_rows_total: Optional[int]
    acc_unit: str = "m/s^2"
    gyr_unit: str = "rad/s"
    mag_unit: str = "uT"


def build_config() -> ProcessingConfig:
    ca_cfg = get_ca_config()
    raw_root = Path(settings.data_storage_path)
    processed_root = Path(getattr(settings, "processed_data_path", Path("./data_storage/processed_data")))
    cpu_count = os.cpu_count() or 1
    requested_workers = int(ca_cfg.processing.workers)
    workers = min(max(1, requested_workers), cpu_count)

    # Prefer `server/ca_config.toml` defaults; allow env/.env to override via the
    # existing Pydantic Settings env var names for backwards compatibility.
    sampling_rate_hz = int(os.environ.get("PROCESSING_SAMPLING_RATE", ca_cfg.windows.sampling_rate_hz))
    min_total_mb = int(os.environ.get("PROCESSING_MIN_TOTAL_MB", ca_cfg.processing.min_total_mb))
    target_total_mb = int(os.environ.get("PROCESSING_TARGET_MB", ca_cfg.processing.target_total_mb))

    def _cap_rows(value: int) -> Optional[int]:
        value = int(value)
        if value <= 0:
            return None
        return value

    hmog_max_rows_per_subject = _cap_rows(
        int(os.environ.get("PROCESSING_HMOG_MAX_ROWS_PER_SUBJECT", ca_cfg.processing.hmog_max_rows_per_subject))
    )
    hmog_max_rows_total = _cap_rows(int(os.environ.get("PROCESSING_HMOG_MAX_ROWS_TOTAL", ca_cfg.processing.hmog_max_rows_total)))

    def _positive_int(value: int) -> int:
        return max(0, int(value))

    hmog_val_subject_count = _positive_int(
        int(os.environ.get("PROCESSING_HMOG_VAL_SUBJECT_COUNT", ca_cfg.processing.hmog_val_subject_count))
    )
    hmog_test_subject_count = _positive_int(
        int(os.environ.get("PROCESSING_HMOG_TEST_SUBJECT_COUNT", ca_cfg.processing.hmog_test_subject_count))
    )
    return ProcessingConfig(
        raw_root=raw_root,
        processed_root=processed_root,
        zscore_root=processed_root / "z-score",
        window_root=processed_root / "window",
        hmog_root=Path(getattr(settings, "hmog_data_path", "/data/code/ca/refer/ContinAuth/src/data/processed/raw_hmog_data")),
        sampling_rate_hz=sampling_rate_hz,
        window_sizes=list(ca_cfg.windows.sizes),
        window_overlap=float(ca_cfg.windows.overlap),
        min_total_bytes=to_bytes(min_total_mb),
        target_total_bytes=to_bytes(target_total_mb),
        workers=workers,
        window_workers=workers,
        hmog_val_subject_count=hmog_val_subject_count,
        hmog_test_subject_count=hmog_test_subject_count,
        hmog_max_rows_per_subject=hmog_max_rows_per_subject,
        hmog_max_rows_total=hmog_max_rows_total,
        acc_unit=getattr(settings, "hmog_acc_unit", "m/s^2"),
        gyr_unit=getattr(settings, "hmog_gyr_unit", "rad/s"),
        mag_unit=getattr(settings, "hmog_mag_unit", "uT"),
    )


def to_bytes(mb: int) -> int:
    return int(mb * 1024 * 1024)


def _half_cores() -> int:
    cores = os.cpu_count() or 2
    return max(1, cores // 2)


_FILENAME_TIMESTAMP_RE = re.compile(r"(\d{10,})")


def _parse_timestamp_ms_from_stem(stem: str) -> Optional[int]:
    matches = _FILENAME_TIMESTAMP_RE.findall(stem)
    if not matches:
        return None
    try:
        value = int(matches[-1])
    except ValueError:
        return None

    # Heuristics: interpret large integers as epoch timestamps.
    # - ns: 19 digits (>=1e17)
    # - us: 16 digits (>=1e14)
    # - ms: 13 digits (>=1e11)
    # - s:  10 digits (>=1e9)
    if value >= 100_000_000_000_000_000:  # ns
        return value // 1_000_000
    if value >= 100_000_000_000_000:  # us
        return value // 1_000
    if value >= 100_000_000_000:  # ms
        return value
    if value >= 1_000_000_000:  # s
        return value * 1000
    return None


def _sort_key(path: Path) -> Tuple[int, str]:
    stem = path.stem
    ts_ms = _parse_timestamp_ms_from_stem(stem)
    if ts_ms is None:
        ts_ms = int(path.stat().st_mtime * 1000)
    return ts_ms, path.name


def _list_sessions_for_user(user_dir: Path) -> List[Path]:
    return sorted([p for p in user_dir.glob("*.jsonl") if p.is_file()], key=_sort_key)


def _select_sessions_for_processing(files: Sequence[Path], cfg: ProcessingConfig) -> List[Path]:
    total_bytes = sum(f.stat().st_size for f in files)
    if not files:
        return []
    if total_bytes < cfg.min_total_bytes:
        logger.info("Skip user %s: total %.2f MB < threshold %.2f MB", files[0].parent.name if files else "unknown", total_bytes / 1e6, cfg.min_total_bytes / 1e6)
        return []

    selected: List[Path] = []
    accumulated = 0
    # Requirement: sort sessions ascending by filename timestamp/mtime, then take
    # the first X sessions whose total size reaches the target.
    for f in files:
        selected.append(f)
        accumulated += f.stat().st_size
        if accumulated >= cfg.target_total_bytes:
            break
    logger.info(
        "Selected %s sessions for user %s (%.2f MB of %.2f MB total)",
        len(selected),
        files[0].parent.name if files else "unknown",
        accumulated / 1e6,
        total_bytes / 1e6,
    )
    return selected


def _read_session_file(session_path: Path) -> List[Dict]:
    packets: List[Dict] = []
    with session_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                packets.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Invalid JSON line in %s", session_path)
    return packets


def _extract_sensor_records(packets: Iterable[Dict]) -> Dict[str, List[Dict]]:
    records: Dict[str, List[Dict]] = {"acc": [], "gyr": [], "mag": []}
    for pkt in packets:
        samples = pkt.get("sensor_data") or pkt.get("sensor_batch", {}).get("samples") or []
        for sample in samples:
            sensor_type = (
                sample.get("sensor_type")
                or sample.get("sensor_name")
                or sample.get("type")
            )
            # Normalize string identifiers
            if isinstance(sensor_type, str):
                sensor_type_norm = sensor_type.strip().lower()
                if sensor_type_norm in {"accelerometer", "acc"}:
                    sensor_type = "accelerometer"
                elif sensor_type_norm in {"gyroscope", "gyr", "gyro"}:
                    sensor_type = "gyroscope"
                elif sensor_type_norm in {"magnetometer", "mag"}:
                    sensor_type = "magnetometer"
            prefix = SENSOR_TYPE_TO_PREFIX.get(sensor_type)
            if prefix not in records:
                continue
            ts_ns = (
                sample.get("timestamp_ns")
                or sample.get("event_timestamp_ns")
                or sample.get("timestamp")
            )
            if ts_ns is None:
                continue
            try:
                ts_ns_int = int(ts_ns)
            except (TypeError, ValueError):
                continue
            ts_ms = int(ts_ns_int // 1_000_000)
            values = sample.get("values")
            if values and isinstance(values, dict):
                x, y, z = values.get("x"), values.get("y"), values.get("z")
            else:
                x, y, z = sample.get("x"), sample.get("y"), sample.get("z")
            if x is None or y is None or z is None:
                continue
            records[prefix].append({"timestamp": ts_ms, "x": float(x), "y": float(y), "z": float(z)})
    return records


def _resample_records(records: Dict[str, List[Dict]], session_label: str, user_id: str, cfg: ProcessingConfig) -> Optional[pd.DataFrame]:
    base_ms = int(1000 / cfg.sampling_rate_hz)
    all_ts: List[int] = []
    for sensor_values in records.values():
        if sensor_values:
            all_ts.extend([r["timestamp"] for r in sensor_values])
    if not all_ts:
        logger.warning("Session %s for user %s has no sensor timestamps", session_label, user_id)
        return None

    min_ts, max_ts = min(all_ts), max(all_ts)
    if min_ts == max_ts:
        logger.warning("Session %s for user %s has a single timestamp", session_label, user_id)
        return None

    # Important: pandas `resample()` aligns bins to the epoch by default. If we
    # build a base_index starting at an arbitrary (non-10ms-aligned) timestamp,
    # reindexing will produce all-NaN rows and the session becomes empty after
    # `dropna()`. We therefore floor/ceil to the 10ms grid so that:
    #   base_index ∈ { ..., t0, t0+10ms, ... } matches resample bins.
    start_ms = (int(min_ts) // base_ms) * base_ms
    end_ms = ((int(max_ts) + base_ms - 1) // base_ms) * base_ms

    base_index = pd.date_range(
        start=pd.to_datetime(start_ms, unit="ms"),
        end=pd.to_datetime(end_ms, unit="ms"),
        freq=f"{base_ms}ms",
    )

    combined = pd.DataFrame(index=base_index)
    for sensor in ("acc", "gyr", "mag"):
        data = records.get(sensor, [])
        if not data:
            logger.warning("Missing %s data in session %s (user %s)", sensor, session_label, user_id)
            return None
        df = pd.DataFrame(data)
        df = df.drop_duplicates(subset="timestamp")
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp_dt")[["x", "y", "z"]].sort_index()
        resampled = df.resample(f"{base_ms}ms").mean().interpolate(method="time", limit_direction="both")
        aligned = resampled.reindex(base_index).interpolate(method="time", limit_direction="both")
        combined[f"{sensor}_x"] = aligned["x"]
        combined[f"{sensor}_y"] = aligned["y"]
        combined[f"{sensor}_z"] = aligned["z"]

    combined = combined.dropna()
    if combined.empty:
        logger.warning("Resampled data empty for session %s (user %s)", session_label, user_id)
        return None

    combined = combined.reset_index(drop=False).rename(columns={"index": "timestamp"})
    combined["timestamp"] = combined["timestamp"].astype("int64") // 1_000_000
    combined.insert(0, "subject", user_id)
    combined.insert(1, "session", session_label)
    combined = combined[
        ["subject", "session", "timestamp"]
        + [f"{prefix}_{axis}" for prefix in ("acc", "gyr", "mag") for axis in ("x", "y", "z")]
    ]
    return combined


def _load_and_resample_session(session_path: Path, cfg: ProcessingConfig) -> Optional[pd.DataFrame]:
    user_id = session_path.parent.name
    session_label = session_path.stem
    packets = _read_session_file(session_path)
    records = _extract_sensor_records(packets)
    return _resample_records(records, session_label, user_id, cfg)


def _assemble_user_dataframe(user_id: str, session_files: Sequence[Path], cfg: ProcessingConfig) -> Optional[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    session_order = {session_path.stem: idx for idx, session_path in enumerate(session_files)}
    try:
        max_workers = min(cfg.workers, len(session_files))
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(_load_and_resample_session, session_path, cfg): session_path for session_path in session_files}
                for future in as_completed(futures):
                    df = future.result()
                    if df is not None:
                        frames.append(df)
        else:
            for session_path in session_files:
                df = _load_and_resample_session(session_path, cfg)
                if df is not None:
                    frames.append(df)
    except Exception as exc:
        logger.warning("Parallel session processing failed for user %s, falling back to serial (%s)", user_id, exc)
        frames = []
        for session_path in session_files:
            df = _load_and_resample_session(session_path, cfg)
            if df is not None:
                frames.append(df)

    if not frames:
        logger.warning("No usable sessions for user %s", user_id)
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["session_order"] = combined["session"].map(session_order).fillna(len(session_order)).astype(int)
    combined = combined.sort_values(["session_order", "timestamp"]).reset_index(drop=True)
    combined = combined.drop(columns=["session_order"])
    return combined


def _split_by_ratio(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split a user's time-series into train/val/test by time length.

    Requirement: split by the user's total time length (≈ row count after 100Hz
    resampling) into:
      - train: 75%
      - val: 12.5%
      - test: 12.5%

    Notes:
      - A raw file is a session; windowing must not cross sessions. We enforce
        this later during window generation by grouping on (subject, session).
      - This splitter may cut within a session (a session can appear in multiple
        splits), which is acceptable per the spec and keeps the ratios accurate.
    """
    if df.empty:
        empty = df.copy()
        return {"train": empty, "val": empty, "test": empty}

    total = len(df)
    if total < 3:
        empty = df.iloc[0:0].copy()
        return {"train": df.reset_index(drop=True), "val": empty, "test": empty}

    train_end = int(total * 0.75)
    val_end = int(total * 0.875)  # 75% + 12.5%

    # Ensure each split has at least one row when possible.
    train_end = max(1, min(train_end, total - 2))
    val_end = max(train_end + 1, min(val_end, total - 1))

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)
    return {"train": train_df, "val": val_df, "test": test_df}


def _write_split_csvs(splits: Dict[str, pd.DataFrame], base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        target = base_dir / f"{name}.csv"
        df.to_csv(target, index=False)
        logger.info("Wrote %s (%d rows)", target, len(df))


def _load_hmog_attackers(subject_ids: Sequence[str], cfg: ProcessingConfig) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    total_rows = 0
    max_total = cfg.hmog_max_rows_total
    max_per_subject = cfg.hmog_max_rows_per_subject
    for subject_id in subject_ids:
        if max_total is not None and total_rows >= max_total:
            break
        subject_dir = cfg.hmog_root / subject_id
        if not subject_dir.exists():
            continue
        subject_rows = 0
        for split in ("train", "val", "test"):
            if max_total is not None and total_rows >= max_total:
                break
            if max_per_subject is not None and subject_rows >= max_per_subject:
                break
            path = subject_dir / f"{subject_id}_{split}.csv"
            if not path.exists():
                continue
            try:
                nrows: Optional[int] = None
                if max_per_subject is not None:
                    nrows = max(int(max_per_subject - subject_rows), 0)
                if max_total is not None:
                    remaining_total = max(int(max_total - total_rows), 0)
                    nrows = remaining_total if nrows is None else min(nrows, remaining_total)
                if nrows is not None and nrows <= 0:
                    continue

                df = pd.read_csv(path, nrows=nrows)
                df = _coerce_hmog_schema(df)
                df = _convert_units(df, cfg)
                frames.append(df)
                subject_rows += int(len(df))
                total_rows += int(len(df))
            except Exception as exc:
                logger.warning("Skip HMOG %s (%s): %s", subject_id, path.name, exc)
                continue

    if not frames:
        return pd.DataFrame(columns=["subject", "session", "timestamp"] + FEATURE_COLUMNS)

    attackers = pd.concat(frames, ignore_index=True)
    attackers = attackers.sort_values(["subject", "session", "timestamp"])
    return attackers


def _convert_units(df: pd.DataFrame, cfg: ProcessingConfig) -> pd.DataFrame:
    converted = df.copy()
    if cfg.acc_unit.lower() in {"g", "gravity"}:
        converted[["acc_x", "acc_y", "acc_z"]] = converted[["acc_x", "acc_y", "acc_z"]] * 9.80665
    if cfg.gyr_unit.lower() in {"deg/s", "degrees/s", "degree/s"}:
        converted[["gyr_x", "gyr_y", "gyr_z"]] = np.deg2rad(converted[["gyr_x", "gyr_y", "gyr_z"]])
    mag_unit = cfg.mag_unit.lower()
    if mag_unit in {"gauss", "gs"}:
        converted[["mag_x", "mag_y", "mag_z"]] = converted[["mag_x", "mag_y", "mag_z"]] * 100.0
    elif mag_unit in {"mt", "millitesla"}:
        converted[["mag_x", "mag_y", "mag_z"]] = converted[["mag_x", "mag_y", "mag_z"]] * 1000.0
    elif mag_unit in {"t", "tesla"}:
        converted[["mag_x", "mag_y", "mag_z"]] = converted[["mag_x", "mag_y", "mag_z"]] * 1_000_000.0
    return converted


def _normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)
    name = re.sub(r"[^a-z0-9_]+", "", name)
    name = re.sub(r"_+", "_", name)
    return name


_HMOG_ALIASES = {
    "subject": "subject",
    "user": "subject",
    "user_id": "subject",
    "userid": "subject",
    "session": "session",
    "session_id": "session",
    "sessionid": "session",
    "timestamp": "timestamp",
    "time": "timestamp",
    "ts": "timestamp",
    "accx": "acc_x",
    "accy": "acc_y",
    "accz": "acc_z",
    "gyrx": "gyr_x",
    "gyry": "gyr_y",
    "gyrz": "gyr_z",
    "magx": "mag_x",
    "magy": "mag_y",
    "magz": "mag_z",
}


def _coerce_hmog_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure HMOG CSV matches server schema before merging."""
    if df.empty:
        return pd.DataFrame(columns=["subject", "session", "timestamp"] + FEATURE_COLUMNS)

    rename_map: Dict[str, str] = {}
    for col in df.columns:
        normalized = _normalize_column_name(col)
        rename_map[col] = _HMOG_ALIASES.get(normalized, normalized)
    df = df.rename(columns=rename_map)

    required = ["subject", "session", "timestamp"] + FEATURE_COLUMNS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"HMOG CSV missing required columns: {missing}")

    out = df[required].copy()
    out["subject"] = out["subject"].astype(str)
    out["session"] = out["session"].astype(str)
    return out


def _compute_scaler(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    features = train_df[FEATURE_COLUMNS]
    mean = features.mean()
    # Match sklearn's StandardScaler (population std, ddof=0).
    std = features.std(ddof=0)
    std_replaced = std.replace(0, 1.0)
    return {"mean": mean.to_dict(), "std": std_replaced.to_dict()}


def _apply_scaler(df: pd.DataFrame, scaler: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return apply_scaler(df, scaler)


def _write_scaler(scaler: Dict[str, Dict[str, float]], target_dir: Path) -> None:
    target = write_scaler(scaler, target_dir)
    logger.info("Saved scaler to %s", target)


def _write_windows(
    df: pd.DataFrame,
    output_path: Path,
    window_size: float,
    sampling_rate: int,
    overlap: float,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    if df.empty:
        pd.DataFrame(columns=df.columns.tolist() + ["window_id"]).to_csv(output_path, index=False)
        return 0

    window_points = max(1, int(round(window_size * sampling_rate)))
    stride_ratio = 1.0 - float(overlap)
    if stride_ratio <= 0.0:
        raise ValueError(f"Invalid overlap={overlap}; expected < 1.0")
    step_points = max(1, int(round(window_points * stride_ratio)))

    batch: List[pd.DataFrame] = []
    batch_rows = 0
    wrote_header = False
    total_rows = 0
    window_id = 0

    for (subject, session), group in df.groupby(["subject", "session"], sort=False):
        group = group.sort_values("timestamp").reset_index(drop=True)
        if len(group) < window_points:
            continue
        start = 0
        while start + window_points <= len(group):
            window_df = group.iloc[start : start + window_points].copy()
            window_df["window_id"] = window_id
            batch.append(window_df)
            batch_rows += len(window_df)
            if batch_rows >= WINDOW_BATCH_ROWS:
                batch_df = pd.concat(batch, ignore_index=True)
                batch_df.to_csv(output_path, mode="a", header=not wrote_header, index=False)
                wrote_header = True
                total_rows += len(batch_df)
                batch.clear()
                batch_rows = 0
            window_id += 1
            start += step_points

    if batch:
        batch_df = pd.concat(batch, ignore_index=True)
        batch_df.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total_rows += len(batch_df)

    if not wrote_header:
        pd.DataFrame(columns=df.columns.tolist() + ["window_id"]).to_csv(output_path, index=False)
        return 0

    return total_rows


def _window_task(input_path: Path, output_path: Path, window_size: float, sampling_rate: int, overlap: float) -> int:
    if not input_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)
        return 0
    df = pd.read_csv(input_path, dtype={"subject": str, "session": str}, low_memory=False)
    return _write_windows(df, output_path, window_size, sampling_rate, overlap)


def _window_split_task(
    input_path: Path,
    output_root: Path,
    user_id: str,
    split: str,
    window_sizes: Sequence[float],
    sampling_rate: int,
    overlap: float,
) -> List[Tuple[float, int]]:
    """
    Generate all window sizes for a single split in one process.

    This avoids re-reading very large CSVs once per window size (which can be
    both slow and memory-heavy when done in parallel).
    """
    df = pd.read_csv(input_path, dtype={"subject": str, "session": str}, low_memory=False)
    results: List[Tuple[float, int]] = []
    for window_size in window_sizes:
        window_dir = f"{float(window_size):.1f}"
        output_path = output_root / window_dir / user_id / f"{split}.csv"
        count = _write_windows(df, output_path, float(window_size), sampling_rate, overlap)
        results.append((float(window_size), int(count)))
    return results


def _generate_windows_for_user(user_id: str, cfg: ProcessingConfig) -> None:
    zscore_dir = cfg.zscore_root / user_id
    split_tasks: List[Tuple[str, Path]] = []
    for split in ("train", "val", "test"):
        input_path = zscore_dir / f"{split}.csv"
        if input_path.exists():
            split_tasks.append((split, input_path))

    if not split_tasks:
        return

    max_workers = min(cfg.window_workers, 5, len(split_tasks))
    if max_workers <= 1:
        for split, input_path in split_tasks:
            try:
                results = _window_split_task(
                    input_path,
                    cfg.window_root,
                    user_id,
                    split,
                    cfg.window_sizes,
                    cfg.sampling_rate_hz,
                    cfg.window_overlap,
                )
                for window_size, count in results:
                    output_path = cfg.window_root / f"{window_size:.1f}" / user_id / f"{split}.csv"
                    logger.info("Window %.1fs %s -> %s (%d rows)", window_size, split, output_path, count)
            except Exception as exc:
                logger.error("Window split task failed for %s %s: %s", user_id, split, exc)
        return

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(
                    _window_split_task,
                    input_path,
                    cfg.window_root,
                    user_id,
                    split,
                    cfg.window_sizes,
                    cfg.sampling_rate_hz,
                    cfg.window_overlap,
                ): (split, input_path)
                for split, input_path in split_tasks
            }
            for future in as_completed(future_map):
                split, _ = future_map[future]
                try:
                    results = future.result()
                    for window_size, count in results:
                        output_path = cfg.window_root / f"{window_size:.1f}" / user_id / f"{split}.csv"
                        logger.info("Window %.1fs %s -> %s (%d rows)", window_size, split, output_path, count)
                except Exception as exc:
                    logger.error("Window split task failed for %s %s: %s", user_id, split, exc)
    except Exception as exc:
        logger.warning("Window parallelization failed for user %s; falling back to sequential (%s)", user_id, exc)
        for split, input_path in split_tasks:
            try:
                results = _window_split_task(
                    input_path,
                    cfg.window_root,
                    user_id,
                    split,
                    cfg.window_sizes,
                    cfg.sampling_rate_hz,
                    cfg.window_overlap,
                )
                for window_size, count in results:
                    output_path = cfg.window_root / f"{window_size:.1f}" / user_id / f"{split}.csv"
                    logger.info("Window %.1fs %s -> %s (%d rows)", window_size, split, output_path, count)
            except Exception as inner_exc:
                logger.error("Window split task failed for %s %s: %s", user_id, split, inner_exc)


def process_user(user_id: str, cfg: Optional[ProcessingConfig] = None) -> None:
    cfg = cfg or build_config()
    user_dir = cfg.raw_root / user_id
    if not user_dir.exists():
        logger.info("User directory %s missing, skipping", user_dir)
        return

    session_files = _list_sessions_for_user(user_dir)
    selected_sessions = _select_sessions_for_processing(session_files, cfg)
    if not selected_sessions:
        return

    user_df = _assemble_user_dataframe(user_id, selected_sessions, cfg)
    if user_df is None or user_df.empty:
        logger.warning("No processed data for user %s", user_id)
        return

    splits = _split_by_ratio(user_df)
    if splits["train"].empty:
        logger.warning("User %s training split is empty, skipping", user_id)
        return

    hmog_subjects = _list_hmog_subjects(cfg)
    val_count = max(0, int(cfg.hmog_val_subject_count))
    test_count = max(0, int(cfg.hmog_test_subject_count))
    attacker_val_ids = hmog_subjects[:val_count] if val_count else []
    attacker_test_ids = hmog_subjects[val_count : val_count + test_count] if test_count else []
    if (val_count + test_count) > len(hmog_subjects):
        logger.warning(
            "HMOG subject dirs=%d < requested val=%d + test=%d; val_ids=%s test_ids=%s",
            len(hmog_subjects),
            val_count,
            test_count,
            attacker_val_ids,
            attacker_test_ids,
        )
    val_attackers = _load_hmog_attackers(attacker_val_ids, cfg)
    test_attackers = _load_hmog_attackers(attacker_test_ids, cfg)
    logger.info(
        "Loaded HMOG attackers (selected: val_ids=%s test_ids=%s; caps: per_subject=%s rows, total=%s rows): val=%d rows (%d subjects), test=%d rows (%d subjects)",
        attacker_val_ids,
        attacker_test_ids,
        "unlimited" if cfg.hmog_max_rows_per_subject is None else int(cfg.hmog_max_rows_per_subject),
        "unlimited" if cfg.hmog_max_rows_total is None else int(cfg.hmog_max_rows_total),
        len(val_attackers),
        len(attacker_val_ids),
        len(test_attackers),
        len(attacker_test_ids),
    )

    splits["val"] = pd.concat([splits["val"], val_attackers], ignore_index=True)
    splits["test"] = pd.concat([splits["test"], test_attackers], ignore_index=True)

    user_processed_dir = cfg.processed_root / user_id
    _write_split_csvs(splits, user_processed_dir)

    scaler = _compute_scaler(splits["train"])
    _write_scaler(scaler, cfg.zscore_root / user_id)

    normalized = {name: _apply_scaler(df, scaler) for name, df in splits.items()}
    _write_split_csvs(normalized, cfg.zscore_root / user_id)

    _generate_windows_for_user(user_id, cfg)


def process_all_users(cfg: Optional[ProcessingConfig] = None) -> None:
    cfg = cfg or build_config()
    if not cfg.raw_root.exists():
        logger.warning("Raw data root %s does not exist", cfg.raw_root)
        return
    for user_dir in sorted([p for p in cfg.raw_root.iterdir() if p.is_dir()]):
        process_user(user_dir.name, cfg)


def _list_hmog_subjects(cfg: ProcessingConfig) -> List[str]:
    if not cfg.hmog_root.exists():
        logger.warning("HMOG data path %s missing", cfg.hmog_root)
        return []
    subject_dirs = [d for d in cfg.hmog_root.iterdir() if d.is_dir()]
    names = [d.name for d in subject_dirs]
    try:
        return sorted(names, key=lambda x: int(x))
    except ValueError:
        return sorted(names)


def main() -> None:
    cfg = build_config()
    process_all_users(cfg)


if __name__ == "__main__":
    main()
