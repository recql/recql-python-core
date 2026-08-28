"""OpenAPI schema drift check (Part 5 / Phase F).

Pins a content hash of the RankQueryConfig-related OpenAPI slice so CI fails
when the SoT moves without an intentional IR update.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

# Side-engine OpenAPI SoT (may be absent on machines without the nextgres checkout).
OPENAPI = Path("/Users/jharris/Development/nextgres/side/side-engine/api/openapi.yaml")
PIN_FILE = Path(__file__).resolve().parent / "openapi_rankquery_hash.txt"


def _hash_openapi_rank_slice(text: str) -> str:
    # Hash the whole file; IR is a closure of many schemas.
    return hashlib.sha256(text.encode()).hexdigest()[:32]


def test_openapi_hash_pin():
    if not OPENAPI.is_file():
        pytest.skip("side-engine openapi.yaml not available on this machine")
    text = OPENAPI.read_text(encoding="utf-8")
    digest = _hash_openapi_rank_slice(text)
    assert PIN_FILE.is_file(), "missing openapi_rankquery_hash.txt pin"
    pinned = PIN_FILE.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    assert digest == pinned, (
        f"OpenAPI SoT drifted: got {digest}, pinned {pinned}. "
        "Update openapi_ir/ models and refresh openapi_rankquery_hash.txt."
    )
