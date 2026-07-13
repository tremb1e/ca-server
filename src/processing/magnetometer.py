from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd


SIX_AXIS_COLUMNS: Tuple[str, ...] = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
)
MAGNETOMETER_COLUMNS: Tuple[str, ...] = ("mag_x", "mag_y", "mag_z")
NINE_AXIS_COLUMNS: Tuple[str, ...] = SIX_AXIS_COLUMNS + MAGNETOMETER_COLUMNS
SUPPORTED_INPUT_HEIGHTS = (6, 9)
SENSOR_MODE_FILENAME = "sensor_mode.json"


def axis_columns(input_height: int) -> Tuple[str, ...]:
    height = int(input_height)
    if height == 6:
        return SIX_AXIS_COLUMNS
    if height == 9:
        return NINE_AXIS_COLUMNS
    raise ValueError(f"Unsupported sensor input_height={height}; expected one of {SUPPORTED_INPUT_HEIGHTS}")


def assess_magnetometer(
    df: pd.DataFrame,
    *,
    max_magnitude_ut: float,
    max_outlier_ratio: float,
    min_coverage_ratio: float,
    min_samples: int,
) -> Dict[str, Any]:
    """Choose a stable per-user 6/9-axis mode from the genuine training source.

    The fixed physical threshold catches strong environmental interference while
    ``max_outlier_ratio`` prevents a few transport/calibration spikes from
    downgrading an otherwise healthy user's complete model.
    """
    threshold = float(max_magnitude_ut)
    allowed_ratio = float(max_outlier_ratio)
    required_coverage = float(min_coverage_ratio)
    required_samples = int(min_samples)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("magnetometer_max_magnitude_ut must be finite and > 0")
    if not 0.0 <= allowed_ratio <= 1.0:
        raise ValueError("magnetometer_max_outlier_ratio must be in [0, 1]")
    if not 0.0 <= required_coverage <= 1.0:
        raise ValueError("magnetometer_min_coverage_ratio must be in [0, 1]")
    if required_samples < 1:
        raise ValueError("magnetometer_min_samples must be >= 1")

    total_rows = int(len(df))
    missing_columns = [col for col in MAGNETOMETER_COLUMNS if col not in df.columns]
    if missing_columns:
        finite_count = 0
        magnitudes = np.empty((0,), dtype=np.float64)
    else:
        values = df[list(MAGNETOMETER_COLUMNS)].to_numpy(dtype=np.float64, copy=False)
        finite_mask = np.isfinite(values).all(axis=1)
        finite_count = int(finite_mask.sum())
        magnitudes = np.linalg.norm(values[finite_mask], axis=1) if finite_count else np.empty((0,), dtype=np.float64)

    coverage_ratio = float(finite_count / total_rows) if total_rows else 0.0
    outlier_count = int((magnitudes > threshold).sum()) if finite_count else 0
    outlier_ratio = float(outlier_count / finite_count) if finite_count else 0.0
    reasons = []
    if missing_columns:
        reasons.append(f"missing_columns={missing_columns}")
    if finite_count < required_samples:
        reasons.append(f"finite_samples={finite_count} < {required_samples}")
    if coverage_ratio < required_coverage:
        reasons.append(f"coverage_ratio={coverage_ratio:.6f} < {required_coverage:.6f}")
    if outlier_ratio > allowed_ratio:
        reasons.append(f"outlier_ratio={outlier_ratio:.6f} > {allowed_ratio:.6f}")

    input_height = 6 if reasons else 9
    quantiles: Dict[str, float] = {}
    if finite_count:
        for q in (0.5, 0.95, 0.99, 0.999):
            quantiles[str(q)] = float(np.quantile(magnitudes, q))

    return {
        "schema_version": 1,
        "input_height": int(input_height),
        "sensor_mode": "acc_gyr" if input_height == 6 else "acc_gyr_mag",
        "feature_columns": list(axis_columns(input_height)),
        "magnetometer_used": bool(input_height == 9),
        "abnormal": bool(input_height == 6),
        "reasons": reasons,
        "method": "magnitude_fixed_threshold_with_outlier_ratio",
        "thresholds": {
            "max_magnitude_ut": threshold,
            "max_outlier_ratio": allowed_ratio,
            "min_coverage_ratio": required_coverage,
            "min_samples": required_samples,
        },
        "statistics": {
            "total_rows": total_rows,
            "finite_samples": finite_count,
            "coverage_ratio": coverage_ratio,
            "outlier_count": outlier_count,
            "outlier_ratio": outlier_ratio,
            "magnitude_min_ut": float(magnitudes.min()) if finite_count else None,
            "magnitude_max_ut": float(magnitudes.max()) if finite_count else None,
            "magnitude_mean_ut": float(magnitudes.mean()) if finite_count else None,
            "magnitude_quantiles_ut": quantiles,
        },
    }


def write_sensor_mode(report: Mapping[str, Any], target_dir: Path) -> Path:
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / SENSOR_MODE_FILENAME
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return target


def load_sensor_mode(path: Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected sensor mode format: {path}")
    height = int(payload.get("input_height", 0))
    axis_columns(height)
    return payload


def sensor_mode_path(processed_root: Path, user_id: str) -> Path:
    return Path(processed_root) / "z-score" / str(user_id) / SENSOR_MODE_FILENAME


def infer_input_height_from_csv(csv_path: Path) -> int:
    """Compatibility fallback for datasets created before sensor_mode.json."""
    import csv

    with Path(csv_path).open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        try:
            header = {str(item).strip().lstrip("\ufeff").lower() for item in next(reader)}
        except StopIteration as exc:
            raise ValueError(f"Empty window CSV: {csv_path}") from exc
    return 9 if set(MAGNETOMETER_COLUMNS).issubset(header) else 6
