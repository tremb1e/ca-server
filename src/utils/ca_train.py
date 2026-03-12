from __future__ import annotations

import sys
from pathlib import Path

from .runtime import app_root, is_frozen


_FROZEN_HELPER_SUBCOMMANDS = {
    "hmog_vqgan_experiment.py": "ca-train-vqgan",
}


def ca_train_root() -> Path:
    return app_root() / "ca_train"


def ensure_ca_train_on_path() -> Path:
    root = ca_train_root()
    if not root.exists():
        if is_frozen():
            return root
        raise FileNotFoundError(f"Missing CA-train directory: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def ca_train_script(name: str) -> Path:
    root = ca_train_root()
    return root / name


def ca_train_command(name: str, *, override: Path | None = None) -> list[str]:
    if override is not None:
        target = Path(override)
        if target.suffix == ".py":
            return [sys.executable, str(target)]
        return [str(target)]

    if is_frozen():
        helper = _FROZEN_HELPER_SUBCOMMANDS.get(name)
        if not helper:
            raise FileNotFoundError(f"No bundled CA-train helper for {name}")
        return [sys.executable, helper]

    script = ca_train_script(name)
    if not script.exists():
        raise FileNotFoundError(f"Missing CA-train script: {script}")
    return [sys.executable, str(script)]
