"""Vector pooling strategies for interaction_pooling and multi-vector query aggregation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


def pool_vectors(
    vectors: Sequence[Sequence[float]],
    *,
    pooling_function: str = "mean",
    weights: Sequence[float] | None = None,
    normalize: bool = False,
) -> list[float]:
    """Pool multiple vectors into a single vector.

    Supported pooling_function values:
    - 'mean' / 'average' / 'avg': arithmetic mean (weighted if weights provided)
    - 'sum': element-wise sum (weighted if weights provided)
    - 'max': element-wise maximum
    - 'min': element-wise minimum

    If vectors is empty, returns an empty list.
    """
    if not vectors:
        return []

    fn = str(pooling_function or "mean").lower().strip()
    w = [float(x) for x in weights] if weights is not None else None
    if w is not None and len(w) != len(vectors):
        raise ValueError(
            f"weights length ({len(w)}) must match vectors length ({len(vectors)})"
        )

    try:
        import numpy as np

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return []

        if fn in ("mean", "average", "avg"):
            if w is not None:
                w_arr = np.array(w, dtype=np.float32)[:, None]
                total_w = np.sum(w_arr)
                if total_w != 0:
                    pooled = np.sum(arr * w_arr, axis=0) / total_w
                else:
                    pooled = np.mean(arr, axis=0)
            else:
                pooled = np.mean(arr, axis=0)
        elif fn == "sum":
            if w is not None:
                w_arr = np.array(w, dtype=np.float32)[:, None]
                pooled = np.sum(arr * w_arr, axis=0)
            else:
                pooled = np.sum(arr, axis=0)
        elif fn == "max":
            pooled = np.max(arr, axis=0)
        elif fn == "min":
            pooled = np.min(arr, axis=0)
        else:
            pooled = np.mean(arr, axis=0)

        if normalize:
            norm = float(np.linalg.norm(pooled))
            if norm > 0:
                pooled = pooled / norm
        return [float(x) for x in pooled]

    except ImportError:
        dims = len(vectors[0])
        n = len(vectors)
        if fn in ("mean", "average", "avg"):
            if w is not None:
                total_w = sum(w) or 1.0
                out = [
                    sum(float(vectors[i][d]) * w[i] for i in range(n)) / total_w
                    for d in range(dims)
                ]
            else:
                out = [
                    sum(float(vectors[i][d]) for i in range(n)) / n
                    for d in range(dims)
                ]
        elif fn == "sum":
            if w is not None:
                out = [
                    sum(float(vectors[i][d]) * w[i] for i in range(n))
                    for d in range(dims)
                ]
            else:
                out = [
                    sum(float(vectors[i][d]) for i in range(n))
                    for d in range(dims)
                ]
        elif fn == "max":
            out = [
                max(float(vectors[i][d]) for i in range(n))
                for d in range(dims)
            ]
        elif fn == "min":
            out = [
                min(float(vectors[i][d]) for i in range(n))
                for d in range(dims)
            ]
        else:
            out = [
                sum(float(vectors[i][d]) for i in range(n)) / n
                for d in range(dims)
            ]

        if normalize:
            norm = math.sqrt(sum(x * x for x in out))
            if norm > 0:
                out = [x / norm for x in out]
        return out
