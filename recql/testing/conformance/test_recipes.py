"""Recipe / CF / pagination conformance (feature-gated)."""

from __future__ import annotations

import os

import pytest

from recql.testing.features import CF, COLUMN_ORDER, PAGINATION, SCORING, VECTOR, require
from recql.testing.testbed import RecqlTestbed


@pytest.mark.asyncio
async def test_recipe_search_sql_yaml(recql_testbed: RecqlTestbed):
    require(recql_testbed, VECTOR)
    sql = """
    SELECT * FROM retrieve(
      text_search(
        query=$query_text, mode='vector', text_embedding_ref='content_embedding',
        name='search', limit=20
      )
    ) LIMIT 10
    """
    yaml_text = recql_testbed.query_fixture("recipe_search.yaml")
    params = {"query_text": "space movies"}
    page_sql = await recql_testbed.execute(sql, params=params)
    page_yaml = await recql_testbed.execute(yaml_text, params=params)
    assert page_sql.ids()
    assert page_sql.ids() == page_yaml.ids()


@pytest.mark.asyncio
async def test_recipe_similar_items(recql_testbed: RecqlTestbed):
    require(recql_testbed, CF)
    yaml_text = recql_testbed.query_fixture("recipe_similar_items.yaml")
    sql = """
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_item(input_item_id=$item_id),
        name='similar', limit=20
      )
    ) LIMIT 10
    """
    params = {"item_id": "3"}
    page_sql = await recql_testbed.execute(sql, params=params)
    page_yaml = await recql_testbed.execute(yaml_text, params=params)
    assert page_sql.ids()
    assert page_sql.ids() == page_yaml.ids()


@pytest.mark.asyncio
async def test_recipe_rerank(recql_testbed: RecqlTestbed):
    require(recql_testbed, SCORING)
    yaml_text = recql_testbed.query_fixture("recipe_rerank.yaml")
    sql = """
    SELECT score(expression='click_through_rate', input_user_id=$user_id) AS ctr
    FROM retrieve(ids(ids=$candidate_item_ids, name='candidates'))
    ORDER BY ctr
    LIMIT 10
    """
    params = {"user_id": "55", "candidate_item_ids": ["1", "3", "5", "6"]}
    page_sql = await recql_testbed.execute(sql, params=params)
    page_yaml = await recql_testbed.execute(yaml_text, params=params)
    assert page_sql.ids()
    assert page_sql.ids() == page_yaml.ids()
    assert "ctr" in page_sql.items[0].attributes


@pytest.mark.asyncio
async def test_recipe_personalized_feed(recql_testbed: RecqlTestbed):
    require(recql_testbed, SCORING)
    yaml_text = recql_testbed.query_fixture("recipe_personalized_feed.yaml")
    page = await recql_testbed.execute(yaml_text, params={"user_id": "55"})
    assert page.ids()
    assert "ctr" in page.items[0].attributes


@pytest.mark.asyncio
async def test_als_u2i(recql_testbed: RecqlTestbed):
    require(recql_testbed, CF)
    sql = """
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_user(input_user_id=$user_id),
        name='user_vector',
        limit=10
      )
    ) LIMIT 5
    """
    page = await recql_testbed.execute(sql, params={"user_id": "55"})
    assert page.ids()


@pytest.mark.asyncio
async def test_als_i2i(recql_testbed: RecqlTestbed):
    require(recql_testbed, CF)
    sql = """
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_item(input_item_id=$item_id),
        name='item_vector',
        limit=10
      )
    ) LIMIT 5
    """
    page = await recql_testbed.execute(sql, params={"item_id": "3"})
    assert page.ids()


@pytest.mark.asyncio
async def test_cold_start_fallback(recql_testbed: RecqlTestbed):
    require(recql_testbed, CF, COLUMN_ORDER)
    col = recql_testbed.popular_rank_column
    sql = f"""
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_user(input_user_id=$user_id),
        name='user_vector',
        limit=10
      ),
      column_order(columns='{col}', name='cold_start', limit=10)
    ) LIMIT 5
    """
    page_warm = await recql_testbed.execute(sql, params={"user_id": "55"})
    assert page_warm.ids()
    for c in page_warm.items:
        if "user_vector" in c.retrieval and c.retrieval["user_vector"] is not None:
            assert c.retrieval_score == c.retrieval["user_vector"]
            break

    page_cold = await recql_testbed.execute(sql, params={"user_id": "ghost"})
    assert page_cold.ids()
    assert any("cold_start" in c.retrieval for c in page_cold.items)


@pytest.mark.asyncio
async def test_pagination_excludes_seen(recql_testbed: RecqlTestbed):
    require(recql_testbed, PAGINATION, COLUMN_ORDER)
    col = recql_testbed.popular_rank_column
    key = f"conf-{recql_testbed.backend}-{os.getpid()}"
    sql = f"""
    SELECT * FROM retrieve(
      column_order(columns='{col} ASC', name='pop', limit=50)
    ) LIMIT 2
    """
    p1 = await recql_testbed.execute(sql, pagination_key=key)
    p2 = await recql_testbed.execute(sql, pagination_key=key)
    assert len(p1.ids()) == 2
    assert len(p2.ids()) == 2
    assert set(p1.ids()).isdisjoint(set(p2.ids()))
