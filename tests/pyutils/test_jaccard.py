"""Jaccard cardinality helpers."""

from __future__ import annotations

from recql.pyutils.jaccard import intersection_cardinality, jaccard


def test_intersection_cardinality_swaps_to_smaller():
    a = {1, 2, 3, 4, 5}
    b = {4, 5, 6}
    assert intersection_cardinality(a, b) == 2
    assert intersection_cardinality(b, a) == 2


def test_jaccard_matches_set_formula():
    a = {"x", "y", "z"}
    b = {"y", "z", "w"}
    # classic: |∩|=2, |∪|=4 → 0.5; cardinality: 2/(3+3-2)=0.5
    assert jaccard(a, b) == 0.5
    assert jaccard(a, a) == 1.0
    assert jaccard(set(), set()) == 0.0
    assert jaccard(a, set()) == 0.0
