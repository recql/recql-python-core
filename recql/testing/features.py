"""Conformance capability flags.

Backends advertise what they implement via ``RecqlTestbed.features``.
Tests skip when a required flag is missing — e.g. FAISS is embedding-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from recql.testing.testbed import RecqlTestbed

# --- capability flags -------------------------------------------------------

SQL_DIALECT = "sql_dialect"
"""Named-bind dialect + bundled SQL ``queries.yaml`` (relational packs only)."""

COLUMN_ORDER = "column_order"
"""Relational ``column_order`` / table scan retrieve."""

VECTOR = "vector"
"""Vector / ANN retrieve (``text_search`` mode=vector, embedding stores)."""

LEXICAL = "lexical"
"""Lexical / full-text retrieve (``text_search`` mode=lexical)."""

HYBRID = "hybrid"
"""Multi-bag lexical + vector merge / RRF. Implies both ``lexical`` and ``vector``."""

CF = "cf"
"""Precomputed embedding similarity (ALS u2i / i2i). Embedding-path; OK for FAISS."""

SCORING = "scoring"
"""Model- or expression-based ``score(...)`` (CTR LightGBM, etc.)."""

PAGINATION = "pagination"
"""KV pagination / seen-item exclusion."""

# Relational SQL backends advertise the full feature set (packs opt in via testbed).
SQL_BACKEND_FEATURES: frozenset[str] = frozenset(
    {
        SQL_DIALECT,
        COLUMN_ORDER,
        VECTOR,
        LEXICAL,
        HYBRID,
        CF,
        SCORING,
        PAGINATION,
    }
)

# FAISS (and similar ANN-only packs): embeddings only — no lexical / scoring / SQL.
FAISS_FEATURES: frozenset[str] = frozenset({VECTOR, CF})

# Document stores (MongoDB, …): full retrieve/score surface without SQL dialect.
DOCUMENT_BACKEND_FEATURES: frozenset[str] = frozenset(
    {
        COLUMN_ORDER,
        VECTOR,
        LEXICAL,
        HYBRID,
        CF,
        SCORING,
        PAGINATION,
    }
)


def require(testbed: "RecqlTestbed", *needed: str) -> None:
    """``pytest.skip`` unless ``testbed.features`` contains every flag in ``needed``."""
    missing = [f for f in needed if f not in testbed.features]
    if missing:
        pytest.skip(
            f"backend {testbed.backend!r} lacks features {missing}; "
            f"has={sorted(testbed.features)}"
        )
