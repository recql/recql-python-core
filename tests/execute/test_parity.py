"""RecQL string vs RankQueryConfig parity."""

from __future__ import annotations

import pytest

from recql.bind import bind
from recql.execute import execute
from recql.harness import recql_to_rank_query_config
from recql.plugins.mock import mock_registry
from recql.yaml_query import query_input_to_rank_query_config


@pytest.mark.asyncio
async def test_sql_and_yaml_same_results():
    sql = """
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_user(input_user_id=$user_id),
        name='uv',
        limit=10
      ),
      column_order(columns='_derived_popular_rank', name='cold', limit=10)
    ) LIMIT 5
    """
    yaml_cfg = {
        "type": "rank",
        "from": "item",
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "als",
                "query_encoder": {
                    "type": "precomputed_user",
                    "input_user_id": "$user_id",
                },
                "name": "uv",
                "limit": 10,
            },
            {
                "type": "column_order",
                "columns": [{"name": "_derived_popular_rank", "ascending": True}],
                "name": "cold",
                "limit": 10,
            },
        ],
        "limit": 5,
    }
    bags = {
        "uv": [("1", 0.9), ("2", 0.8), ("3", 0.7)],
        "cold": [("3", 0.5), ("4", 0.4), ("5", 0.3)],
    }
    reg = mock_registry(bags)

    cfg_sql = query_input_to_rank_query_config(sql)
    cfg_yaml = query_input_to_rank_query_config(yaml_cfg)
    page_sql = await execute(bind(cfg_sql), reg)
    page_yaml = await execute(bind(cfg_yaml), reg)
    assert page_sql.ids() == page_yaml.ids()
    assert [c.retrieval_score for c in page_sql.items] == [
        c.retrieval_score for c in page_yaml.items
    ]

    # both lower/load to same retrieve types
    a = recql_to_rank_query_config(sql)
    b = recql_to_rank_query_config(yaml_cfg)
    assert len(a["retrieve"]) == len(b["retrieve"]) == 2
