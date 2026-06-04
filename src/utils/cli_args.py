from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence


def normalize_option_value_args(
    argv: Sequence[str] | None,
    *,
    options: Iterable[str],
) -> list[str]:
    raw = list(sys.argv[1:] if argv is None else argv)
    option_set = {str(option) for option in options}
    normalized: list[str] = []
    i = 0
    while i < len(raw):
        arg = raw[i]
        if arg in option_set and i + 1 < len(raw) and raw[i + 1].startswith("-"):
            normalized.append(f"{arg}={raw[i + 1]}")
            i += 2
            continue
        normalized.append(arg)
        i += 1
    return normalized
