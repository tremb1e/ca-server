from __future__ import annotations

import sys
from pathlib import Path


def ca_train_root() -> Path:
    # server/src/utils/ca_train.py -> server/ca_train
    return Path(__file__).resolve().parents[2] / "ca_train"


def ensure_ca_train_on_path() -> Path:
    root = ca_train_root()
    if not root.exists():
        raise FileNotFoundError(f"Missing CA-train directory: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ca_train_script(name: str) -> Path:
    root = ca_train_root()
    return root / name
