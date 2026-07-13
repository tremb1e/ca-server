import json

import pandas as pd
import pytest

from src.processing.magnetometer import (
    assess_magnetometer,
    axis_columns,
    load_sensor_mode,
    write_sensor_mode,
)


def _frame(magnitudes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "acc_x": [0.0] * len(magnitudes),
            "acc_y": [0.0] * len(magnitudes),
            "acc_z": [9.8] * len(magnitudes),
            "gyr_x": [0.0] * len(magnitudes),
            "gyr_y": [0.0] * len(magnitudes),
            "gyr_z": [0.0] * len(magnitudes),
            "mag_x": magnitudes,
            "mag_y": [0.0] * len(magnitudes),
            "mag_z": [0.0] * len(magnitudes),
        }
    )


def _assess(df: pd.DataFrame):
    return assess_magnetometer(
        df,
        max_magnitude_ut=120.0,
        max_outlier_ratio=0.01,
        min_coverage_ratio=0.95,
        min_samples=100,
    )


def test_normal_magnetometer_selects_nine_axes() -> None:
    # Exactly 1% threshold exceedance is tolerated (comparison is strictly >).
    report = _assess(_frame([50.0] * 990 + [121.0] * 10))
    assert report["input_height"] == 9
    assert report["magnetometer_used"] is True
    assert report["statistics"]["outlier_ratio"] == pytest.approx(0.01)


def test_environmental_interference_selects_six_axes() -> None:
    report = _assess(_frame([50.0] * 700 + [1800.0] * 300))
    assert report["input_height"] == 6
    assert report["abnormal"] is True
    assert "outlier_ratio" in report["reasons"][-1]


def test_missing_or_sparse_magnetometer_selects_six_axes() -> None:
    df = _frame([50.0] * 100)
    df.loc[:9, ["mag_x", "mag_y", "mag_z"]] = float("nan")
    report = _assess(df)
    assert report["input_height"] == 6
    assert report["statistics"]["coverage_ratio"] == pytest.approx(0.9)


def test_sensor_mode_round_trip_and_axis_validation(tmp_path) -> None:
    report = _assess(_frame([50.0] * 100))
    path = write_sensor_mode(report, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["input_height"] == 9
    assert load_sensor_mode(path)["feature_columns"] == list(axis_columns(9))
    with pytest.raises(ValueError, match="Unsupported"):
        axis_columns(12)
