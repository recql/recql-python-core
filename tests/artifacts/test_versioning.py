"""Artifact pinning — unit tests only (DB registration lives in backend packs)."""

from __future__ import annotations

import pytest

from recql.artifacts import (
    check_embedding_dims,
    check_feature_spec_compat,
    config_hash,
    pins_from_deployment,
    resolve_version,
)
from recql.errors import BindError


def test_config_hash_stable():
    a = config_hash({"name": "x", "dims": 8})
    b = config_hash({"dims": 8, "name": "x"})
    assert a == b
    assert len(a) == 16


def test_pins_and_resolve():
    pins = pins_from_deployment(
        {
            "artifact_version": "v1",
            "model_versions": {"click_through_rate": "v2"},
        }
    )
    assert resolve_version("click_through_rate", pins) == "v2"
    assert resolve_version("other_model", pins) == "v1"


def test_feature_spec_mismatch_raises():
    with pytest.raises(BindError, match="feature_spec"):
        check_feature_spec_compat(
            {"features": ["a", "b"]},
            {"features": ["a", "c"]},
        )


def test_dims_mismatch_raises():
    with pytest.raises(BindError, match="dims"):
        check_embedding_dims(8, 16)
