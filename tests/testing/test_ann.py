"""Unit tests for deterministic ANN helpers (no database)."""

from __future__ import annotations

import pytest

from recql.encode import fake_embedding
from recql.testing.ann import (
    assert_ann_agreement,
    brute_force_vector_ids,
    cosine,
    load_ann_corpus,
    overlap_at_k,
)


def test_fake_embedding_is_deterministic():
    a = fake_embedding("Toy Story (1995)", dims=8)
    b = fake_embedding("Toy Story (1995)", dims=8)
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_brute_force_exact_title_ranks_first():
    corpus = load_ann_corpus()
    ids = brute_force_vector_ids("Toy Story (1995)", corpus, dims=8, k=5)
    assert ids[0] == "1"


def test_overlap_and_agreement_threshold():
    expected = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    # 6/10 overlap
    actual = ["1", "2", "3", "4", "5", "6", "99", "98", "97", "96"]
    assert overlap_at_k(actual, expected, k=10) == pytest.approx(0.6)
    assert_ann_agreement(actual, expected, k=10, min_overlap=0.5)
    with pytest.raises(AssertionError):
        assert_ann_agreement(actual, expected, k=10, min_overlap=0.9)


def test_cosine_identical_is_one():
    v = fake_embedding("x", dims=4)
    assert cosine(v, v) == pytest.approx(1.0)
