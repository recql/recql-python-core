"""RecQL testing helpers — shared conformance suite for backend packs."""

from __future__ import annotations

from recql.testing import features
from recql.testing.ann import assert_ann_agreement, brute_force_vector_ids, overlap_at_k
from recql.testing.features import (
    DOCUMENT_BACKEND_FEATURES,
    FAISS_FEATURES,
    SQL_BACKEND_FEATURES,
    require,
)
from recql.testing.testbed import RecqlTestbed

__all__ = [
    "DOCUMENT_BACKEND_FEATURES",
    "FAISS_FEATURES",
    "SQL_BACKEND_FEATURES",
    "RecqlTestbed",
    "assert_ann_agreement",
    "brute_force_vector_ids",
    "features",
    "overlap_at_k",
    "require",
]
