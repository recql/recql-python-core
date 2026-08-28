"""Stage retrieve + merge conformance (feature-gated per backend)."""

from __future__ import annotations

import pytest

from recql.testing.features import COLUMN_ORDER, HYBRID, VECTOR, require
from recql.testing.testbed import RecqlTestbed


@pytest.mark.asyncio
async def test_stage1_semantic_vector_search(recql_testbed: RecqlTestbed):
    """Exact title under fake encoder → id 1 (deterministic). Needs ``vector``."""
    require(recql_testbed, VECTOR)
    from recql.harness import recql_to_rank_query_config

    sql = """
    SELECT * FROM retrieve(
      text_search(
        query='Toy Story (1995)',
        mode='vector',
        text_embedding_ref='title_embedding',
        name='vec',
        limit=50
      )
    ) LIMIT 5
    """
    yaml_text = recql_testbed.query_fixture("stage1_semantic.yaml")
    assert recql_to_rank_query_config(sql)["retrieve"][0]["type"] == "text_search"

    page_sql = await recql_testbed.execute(sql)
    page_yaml = await recql_testbed.execute(yaml_text)
    assert page_sql.ids()
    assert page_sql.ids() == page_yaml.ids()
    assert page_sql.ids()[0] == "1"


@pytest.mark.asyncio
async def test_stage1_column_order_popular(recql_testbed: RecqlTestbed):
    require(recql_testbed, COLUMN_ORDER)
    col = recql_testbed.popular_rank_column
    sql = f"""
    SELECT * FROM retrieve(
      column_order(columns='{col}', name='pop', limit=10)
    ) LIMIT 3
    """
    page = await recql_testbed.execute(sql)
    assert len(page.ids()) == 3
    expected = await recql_testbed.popular_item_ids(3)
    if expected is not None:
        assert page.ids() == expected


@pytest.mark.asyncio
async def test_stage3_hybrid_merge_stash(recql_testbed: RecqlTestbed):
    require(recql_testbed, HYBRID)
    sql = """
    SELECT * FROM retrieve(
      text_search(query='space movies', mode='lexical', name='lexical', limit=50),
      text_search(query='space movies', mode='vector', text_embedding_ref='content_embedding',
                  name='vector', limit=50)
    ) LIMIT 10
    """
    page = await recql_testbed.execute(sql)
    assert page.ids()
    for c in page.items:
        if "lexical" in c.retrieval and "vector" in c.retrieval:
            assert c.retrieval_score == c.retrieval["lexical"]
            break


@pytest.mark.asyncio
async def test_stage3_hybrid_rrf_sql_yaml(recql_testbed: RecqlTestbed):
    require(recql_testbed, HYBRID)
    sql = """
    SELECT score(
      expression='1.0 / (60 + retrieval.get_rank("vector", 999)) + 1.0 / (60 + retrieval.get_rank("lexical", 999))'
    ) AS fused
    FROM retrieve(
      text_search(query='space movies', mode='lexical', name='lexical', limit=50),
      text_search(query='space movies', mode='vector', text_embedding_ref='content_embedding',
                  name='vector', limit=50)
    )
    ORDER BY fused
    LIMIT 5
    """
    yaml_cfg = {
        "type": "rank",
        "from": "item",
        "retrieve": [
            {
                "type": "text_search",
                "input_text_query": "space movies",
                "mode": {"type": "lexical"},
                "name": "lexical",
                "limit": 50,
            },
            {
                "type": "text_search",
                "input_text_query": "space movies",
                "mode": {"type": "vector", "text_embedding_ref": "content_embedding"},
                "name": "vector",
                "limit": 50,
            },
        ],
        "score": {
            "type": "score_ensemble",
            "value_model": (
                '1.0 / (60 + retrieval.get_rank("vector", 999))'
                ' + 1.0 / (60 + retrieval.get_rank("lexical", 999))'
            ),
            "output_alias": "fused",
            "preserve_order": False,
        },
        "limit": 5,
    }
    page_sql = await recql_testbed.execute(sql)
    page_yaml = await recql_testbed.execute(yaml_cfg)
    assert page_sql.ids()
    assert page_sql.ids() == page_yaml.ids()
    assert "fused" in page_sql.items[0].attributes
