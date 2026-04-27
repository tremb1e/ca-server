from __future__ import annotations

from pathlib import Path
from typing import Any


class UnsafePathSegmentError(ValueError):
    pass


def validate_storage_id(value: Any, *, field_name: str = "id", max_length: int = 255) -> str:
    text = str(value or "")
    if not text:
        raise UnsafePathSegmentError(f"{field_name} must not be empty")
    if len(text) > max_length:
        raise UnsafePathSegmentError(f"{field_name} is too long")
    if text in {".", ".."} or "/" in text or "\\" in text or "\x00" in text:
        raise UnsafePathSegmentError(f"{field_name} contains unsafe path characters")
    return text


def safe_child_path(base_path: Path, *segments: Any) -> Path:
    base = Path(base_path).resolve()
    current = base
    for index, segment in enumerate(segments):
        current = current / validate_storage_id(segment, field_name=f"path segment {index + 1}")
    resolved = current.resolve()
    if resolved != base and base not in resolved.parents:
        raise UnsafePathSegmentError("path escapes base directory")
    return resolved
