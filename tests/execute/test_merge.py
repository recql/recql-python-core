"""§2.2 merge finish-order independence tests."""

from __future__ import annotations

import pytest

from recql.bind import bind
from recql.execute import execute
from recql.execute.merge import Candidate, RetrieveBag, union_dedupe
from recql.openapi_ir import convert_rank_query_config
from recql.plugins.mock import mock_registry


def test_union_first_non_null_not_max_score():
    bags = [
        RetrieveBag(
            "user_vector",
            [Candidate("a", 0.5), Candidate("b", 0.4)],
        ),
        RetrieveBag(
            "cold_start",
            [Candidate("a", 0.99), Candidate("c", 0.9)],
        ),
    ]
    merged = union_dedupe(bags)
    by_id = {c.id: c for c in merged}
    assert by_id["a"].retrieval_score == 0.5  # first bag wins, not 0.99
    assert by_id["a"].retrieval["user_vector"] == 0.5
    assert by_id["a"].retrieval["cold_start"] == 0.99
    assert by_id["a"].retrieval["user_vector_rank"] == 1
    assert by_id["a"].retrieval["cold_start_rank"] == 1
    assert by_id["c"].retrieval_score == 0.9
    assert [c.id for c in merged] == ["a", "b", "c"]  # emission order


@pytest.mark.asyncio
async def test_finish_order_independence():
    """Shuffle completion delays; merge result identical."""
    config = convert_rank_query_config(
        {
            "type": "rank",
            "from": "item",
            "retrieve": [
                {
                    "type": "similarity",
                    "embedding_ref": "als",
                    "query_encoder": {
                        "type": "precomputed_user",
                        "input_user_id": "$u",
                    },
                    "name": "user_vector",
                    "limit": 10,
                },
                {
                    "type": "column_order",
                    "columns": [{"name": "_derived_popular_rank", "ascending": True}],
                    "name": "cold_start",
                    "limit": 10,
                },
            ],
            "limit": 10,
        }
    )
    bags = {
        "user_vector": [("a", 0.5), ("b", 0.4)],
        "cold_start": [("a", 0.99), ("c", 0.9)],
    }
    results = []
    for delays in (
        {"user_vector": 0.01, "cold_start": 0.05},
        {"user_vector": 0.05, "cold_start": 0.01},
        {"user_vector": 0.0, "cold_start": 0.0},
    ):
        reg = mock_registry(bags, delay_by_name=delays)
        bound = bind(config, None, params={"u": "1"})
        page = await execute(bound, reg)
        results.append(
            (
                page.ids(),
                [(c.id, c.retrieval_score, dict(c.retrieval)) for c in page.items],
            )
        )
    assert results[0] == results[1] == results[2]
    assert results[0][0] == ["a", "b", "c"]
    assert results[0][1][0][1] == 0.5


@pytest.mark.asyncio
async def test_prefilter_fail_closed():
    from recql.errors import ExecuteError

    config = convert_rank_query_config(
        {
            "type": "rank",
            "from": "item",
            "retrieve": [
                {
                    "type": "filter",
                    "where": "category = 'x'",
                    "name": "f",
                    "limit": 5,
                }
            ],
            "limit": 5,
        }
    )
    reg = mock_registry({"f": [("a", 1.0)]}, supports_where=False)
    bound = bind(config)
    with pytest.raises(ExecuteError, match="fail closed"):
        await execute(bound, reg)
