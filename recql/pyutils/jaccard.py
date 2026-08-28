"""Set similarity helpers."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from typing import Any


def intersection_cardinality(a: Collection[Any], b: Collection[Any]) -> int:
    """Count |A ∩ B| without allocating an intersection set.

    Iterates the smaller collection and probes membership in the larger.
    Prefer ``b`` (or both) to be a set/dict for O(1) membership.
    """
    if len(a) > len(b):
        a, b = b, a
    # Ensure membership probe is O(1) when possible.
    if not isinstance(b, (set, frozenset, dict)):
        b = set(b)
    return sum(1 for x in a if x in b)


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    """Jaccard index via cardinalities: c / (a + b - c).

    Avoids materializing A∩B and A∪B sets.
    """
    sa = a if isinstance(a, (set, frozenset)) else set(a)
    sb = b if isinstance(b, (set, frozenset)) else set(b)
    if not sa and not sb:
        return 0.0
    if not sa or not sb:
        return 0.0
    ca, cb = len(sa), len(sb)
    c = intersection_cardinality(sa, sb)
    denom = ca + cb - c
    return (c / denom) if denom else 0.0
