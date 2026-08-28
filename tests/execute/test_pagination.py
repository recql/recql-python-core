"""Pagination KV exclusion tests."""

from __future__ import annotations

import pytest

from recql import recql
from recql.plugins.mock import mock_registry


@pytest.mark.asyncio
async def test_pagination_excludes_seen_ids():
    bags = {
        "uv": [("a", 1.0), ("b", 0.9), ("c", 0.8), ("d", 0.7)],
    }
    registry = mock_registry(bags)
    sql = (
        "SELECT * FROM retrieve("
        "similarity(embedding_ref='als', name='uv', limit=10)"
        ") LIMIT 2"
    )
    page1 = await recql(
        query=sql, params={"user_id": "u1"}, backend=registry, pagination_key="sess1"
    )
    assert page1.ids() == ["a", "b"]
    page2 = await recql(
        query=sql, params={"user_id": "u1"}, backend=registry, pagination_key="sess1"
    )
    assert page2.ids() == ["c", "d"]
    assert any(d.startswith("elapsed_ms=") for d in page2.diagnostics)
