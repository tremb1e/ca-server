from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def serialize_policy_path(path: Path, *, relative_to: Path) -> str:
    path = Path(path)
    relative_to = Path(relative_to)
    try:
        rel = os.path.relpath(path, start=relative_to)
    except ValueError:
        return str(path)
    return rel.replace(os.sep, "/")


def resolve_policy_path(
    raw_path: str | Path | None,
    *,
    policy_path: Path,
    server_root: Optional[Path] = None,
    models_root: Optional[Path] = None,
) -> Path:
    if raw_path is None:
        return Path()
    raw_text = str(raw_path).strip()
    if not raw_text:
        return Path()

    policy_path = Path(policy_path)
    candidate = Path(raw_text)
    if not candidate.is_absolute():
        return (policy_path.parent / candidate).resolve(strict=False)

    if candidate.exists():
        return candidate

    rebased = rebase_legacy_policy_path(candidate, server_root=server_root, models_root=models_root)
    if rebased is not None:
        return rebased
    return candidate


def rebase_legacy_policy_path(
    path: Path,
    *,
    server_root: Optional[Path] = None,
    models_root: Optional[Path] = None,
) -> Optional[Path]:
    path = Path(path)
    if not path.is_absolute():
        return None

    parts = path.parts
    for idx in range(len(parts) - 2):
        if parts[idx] == "data_storage" and parts[idx + 1] == "models":
            if models_root is not None:
                return Path(models_root).joinpath(*parts[idx + 2 :]).resolve(strict=False)
            if server_root is not None:
                return Path(server_root).joinpath(*parts[idx:]).resolve(strict=False)
    if server_root is not None:
        for idx in range(len(parts) - 1):
            if parts[idx] == "data_storage":
                return Path(server_root).joinpath(*parts[idx:]).resolve(strict=False)
    if models_root is not None:
        for idx in range(len(parts) - 1):
            if parts[idx] == "models":
                return Path(models_root).joinpath(*parts[idx + 1 :]).resolve(strict=False)
    return None
