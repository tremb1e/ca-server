"""Helpers for command-line argument parsing.

Some device/user identifiers are base64-like hashes that can legitimately start
with a dash (e.g. ``-NMxmSk...``). ``argparse`` treats such a value as an option
flag, which breaks ``--user -NMxmSk...`` style invocations (and the training
subprocess built from them). :func:`normalize_option_value_args` rewrites
``--user <dash-value>`` into ``--user=<dash-value>`` so the value is parsed as a
value rather than an unknown option.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable, Sequence


def normalize_option_value_args(
    argv: Sequence[str] | None,
    *,
    options: Iterable[str],
) -> list[str]:
    """Rewrite ``--opt <dash-prefixed-value>`` into ``--opt=<dash-prefixed-value>``.

    Only the options listed in ``options`` are rewritten, and only when their
    value starts with ``-`` (so normal usage is untouched). ``argv`` of ``None``
    falls back to ``sys.argv[1:]``.
    """

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
