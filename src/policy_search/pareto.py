from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class ParetoPoint:
    """A point in objective space with attached payload."""

    objectives: Tuple[float, ...]
    payload: dict


def pareto_frontier(points: Iterable[ParetoPoint]) -> List[ParetoPoint]:
    """
    Return non-dominated points (minimization) preserving input order.

    A point A dominates B if:
      - A is <= B in all objectives, and
      - A is <  B in at least one objective.
    """
    pts = list(points)
    frontier: List[ParetoPoint] = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i == j:
                continue
            if _dominates(q.objectives, p.objectives):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


def _dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    if len(a) != len(b):
        raise ValueError("Objective dimensions must match")
    le_all = True
    lt_any = False
    for x, y in zip(a, b, strict=True):
        if x > y:
            le_all = False
            break
        if x < y:
            lt_any = True
    return le_all and lt_any

