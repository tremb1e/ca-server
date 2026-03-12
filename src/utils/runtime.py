from __future__ import annotations

import os
import sys
from pathlib import Path


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass).resolve()
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def app_root() -> Path:
    configured = os.getenv("APP_ROOT") or os.getenv("CA_APP_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()

    if is_frozen():
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir.name == "bin":
            return exe_dir.parent
        return exe_dir

    return Path(__file__).resolve().parents[2]


def append_env_pythonpath(*, prepend_bundle_root: bool = True) -> None:
    candidates: list[str] = []

    if prepend_bundle_root:
        candidates.append(str(bundle_root()))

    raw_pythonpath = os.getenv("PYTHONPATH", "")
    if raw_pythonpath:
        candidates.extend(part for part in raw_pythonpath.split(os.pathsep) if part)

    seen = {str(Path(entry).expanduser()) for entry in sys.path if entry}
    for candidate in candidates:
        resolved = str(Path(candidate).expanduser())
        if not resolved or resolved in seen:
            continue
        sys.path.insert(0, resolved)
        seen.add(resolved)
