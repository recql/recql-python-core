"""LightGBM load-once / predict-hot-path."""

from __future__ import annotations

import pytest

from recql.scoring import load_lgbm_booster, predict_lgbm, predict_lgbm_blob


def _tiny_blob() -> bytes:
    import lightgbm as lgb
    import numpy as np

    X = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    y = np.array([0.0, 1.0, 0.5])
    train = lgb.Dataset(X, label=y)
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1, "num_leaves": 3},
        train,
        num_boost_round=5,
    )
    return booster.model_to_string().encode("utf-8")


def test_predict_reuses_loaded_booster():
    pytest.importorskip("lightgbm")
    pytest.importorskip("numpy")
    blob = _tiny_blob()
    booster = load_lgbm_booster(blob)
    a = predict_lgbm(booster, [[0.0, 1.0], [1.0, 0.0]])
    b = predict_lgbm(booster, [[0.0, 1.0], [1.0, 0.0]])
    assert a == b
    assert len(a) == 2
    # one-shot path matches warmed path
    c = predict_lgbm_blob(blob, [[0.0, 1.0], [1.0, 0.0]])
    assert c == a
