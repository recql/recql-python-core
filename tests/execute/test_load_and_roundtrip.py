"""Sibling SQL↔YAML ROUNDTRIP + load smoke (Phase F / Part 17)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from recql.harness import recql, recql_to_rank_query_config
from recql.plugins.mock import mock_registry
from recql.yaml_query import normalize_config_dict

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_recipe_search_sql_yaml_normalized_roundtrip():
    sql = """
    SELECT * FROM retrieve(
      text_search(
        query=$query_text, mode='vector', text_embedding_ref='content_embedding',
        name='search', limit=20
      )
    ) LIMIT 10
    """
    yaml_text = (FIXTURES / "queries" / "recipe_search.yaml").read_text()
    a = normalize_config_dict(recql_to_rank_query_config(sql))
    b = normalize_config_dict(recql_to_rank_query_config(yaml_text))
    assert a["type"] == b["type"] == "rank"
    assert a["retrieve"][0]["type"] == b["retrieve"][0]["type"] == "text_search"
    assert a["retrieve"][0]["mode"]["type"] == b["retrieve"][0]["mode"]["type"]
    assert a["limit"] == b["limit"]


def test_recipe_similar_sql_yaml_normalized_roundtrip():
    sql = """
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_item(input_item_id=$item_id),
        name='similar', limit=20
      )
    ) LIMIT 10
    """
    yaml_text = (FIXTURES / "queries" / "recipe_similar_items.yaml").read_text()
    a = normalize_config_dict(recql_to_rank_query_config(sql))
    b = normalize_config_dict(recql_to_rank_query_config(yaml_text))
    assert a["retrieve"][0]["query_encoder"]["type"] == "precomputed_item"
    assert b["retrieve"][0]["query_encoder"]["type"] == "precomputed_item"


@pytest.mark.asyncio
async def test_concurrent_retrieve_load_smoke():
    """Part 17.7 — concurrent retrieve under load (mock)."""
    bags = {f"b{i}": [(f"i{j}", float(j)) for j in range(20)] for i in range(8)}
    # Map all retrieve types to same bags via names in steps
    reg = mock_registry(
        {"bag": [("a", 1.0), ("b", 0.9), ("c", 0.8)]},
        delay_by_name={"bag": 0.01},
    )

    async def one(n: int):
        sql = (
            "SELECT * FROM retrieve("
            "similarity(embedding_ref='als', name='bag', limit=10)"
            f") LIMIT {1 + (n % 3)}"
        )
        return await recql(query=sql, params={"user_id": "u"}, backend=reg)

    pages = await asyncio.gather(*[one(i) for i in range(40)])
    assert all(p.ids() for p in pages)
    assert all(any(d.startswith("elapsed_ms=") for d in p.diagnostics) for p in pages)


@pytest.mark.asyncio
async def test_timeout_raises():
    from recql.errors import ExecuteError

    reg = mock_registry(
        {"bag": [("a", 1.0)]},
        delay_by_name={"bag": 0.5},
    )
    sql = "SELECT * FROM retrieve(similarity(embedding_ref='als', name='bag', limit=5)) LIMIT 1"
    with pytest.raises(ExecuteError, match="timeout"):
        await recql(query=sql, params={"user_id": "u"}, backend=reg, timeout_s=0.05)
