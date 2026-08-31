import math
import pytest
from recql.encode.pooling import pool_vectors


def test_pool_vectors_empty():
    assert pool_vectors([]) == []


def test_pool_vectors_mean():
    v1 = [1.0, 2.0, 3.0]
    v2 = [3.0, 4.0, 5.0]
    res = pool_vectors([v1, v2], pooling_function="mean")
    assert res == pytest.approx([2.0, 3.0, 4.0])


def test_pool_vectors_sum():
    v1 = [1.0, 2.0, 3.0]
    v2 = [3.0, 4.0, 5.0]
    res = pool_vectors([v1, v2], pooling_function="sum")
    assert res == pytest.approx([4.0, 6.0, 8.0])


def test_pool_vectors_max():
    v1 = [5.0, 1.0, 3.0]
    v2 = [3.0, 4.0, 2.0]
    res = pool_vectors([v1, v2], pooling_function="max")
    assert res == pytest.approx([5.0, 4.0, 3.0])


def test_pool_vectors_min():
    v1 = [5.0, 1.0, 3.0]
    v2 = [3.0, 4.0, 2.0]
    res = pool_vectors([v1, v2], pooling_function="min")
    assert res == pytest.approx([3.0, 1.0, 2.0])


def test_pool_vectors_weighted():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    res = pool_vectors([v1, v2], pooling_function="mean", weights=[3.0, 1.0])
    assert res == pytest.approx([0.75, 0.25])


def test_pool_vectors_normalize():
    v1 = [3.0, 0.0]
    v2 = [0.0, 4.0]
    res = pool_vectors([v1, v2], pooling_function="sum", normalize=True)
    # [3.0, 4.0] normalized has length 5.0 -> [0.6, 0.8]
    assert res == pytest.approx([0.6, 0.8])
