from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

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


def load_scaler(path: Path) -> Dict[str, Dict[str, float]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected scaler format: {path}")
    return payload


def apply_scaler(df: pd.DataFrame, scaler: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    mean = scaler.get("mean", {})
    std = scaler.get("std", {})
    out = df.copy()
    for col in FEATURE_COLUMNS:
        if col not in out.columns:
            continue
        col_mean = float(mean.get(col, 0.0))
        col_std = float(std.get(col, 1.0))
        if col_std == 0:
            col_std = 1.0
        out[col] = (out[col] - col_mean) / col_std
    return out


def write_scaler(scaler: Dict[str, Dict[str, float]], target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "scaler.json"
    target.write_text(json.dumps(scaler, indent=2, ensure_ascii=False), encoding="utf-8")
    return target
