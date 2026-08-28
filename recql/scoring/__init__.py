"""Online scoring helpers — load/predict only (no training).

Call ``load_lgbm_booster`` once (registry warm / first use), then
``predict_lgbm`` on the hot path. Do not re-parse the model string per query.
"""

from __future__ import annotations

from typing import Any


def load_lgbm_booster(blob: bytes | str) -> Any:
    """Deserialize a LightGBM booster from a stored model string / BYTEA."""
    import lightgbm as lgb

    model_str = blob.decode("utf-8") if isinstance(blob, (bytes, bytearray)) else str(blob)
    return lgb.Booster(model_str=model_str)


def predict_lgbm(booster: Any, feature_rows: list[list[float]]) -> list[float]:
    """Run inference with an already-loaded booster."""
    import numpy as np

    X = np.asarray(feature_rows, dtype=float)
    preds = booster.predict(X)
    return [float(p) for p in preds]


def predict_lgbm_blob(blob: bytes, feature_rows: list[list[float]]) -> list[float]:
    """One-shot load + predict (tests / ad-hoc only — prefer warm + ``predict_lgbm``)."""
    return predict_lgbm(load_lgbm_booster(blob), feature_rows)


def _dot(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def click_through_rate_features(
    candidate: Any,
    *,
    user_als: list[float] | None = None,
    item_als: list[float] | None = None,
) -> list[float]:
    """Feature row for demo ``click_through_rate`` — ALS similarity + popularity."""
    als = None
    if hasattr(candidate, "get_score"):
        als = candidate.get_score("user_vector", None)
    if als is None and user_als is not None and item_als is not None:
        als = _dot(user_als, item_als)
    pop = candidate.attributes.get("_derived_popular_rank")
    if pop is None:
        pop = candidate.attributes.get("derived_popular_rank")
    return [
        float(als or 0.0),
        float(pop or 0.0),
        float(len(str(getattr(candidate, "attributes", {}) or {}))),
    ]


__all__ = [
    "click_through_rate_features",
    "load_lgbm_booster",
    "predict_lgbm",
    "predict_lgbm_blob",
]
