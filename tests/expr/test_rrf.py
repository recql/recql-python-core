"""Expression / RRF / retrieval.get_rank tests."""

from __future__ import annotations

import pytest

from recql.bind import bind
from recql.execute import execute
from recql.execute.merge import Candidate
from recql.expr import EvalContext, eval_expr_string
from recql.openapi_ir import convert_rank_query_config
from recql.plugins.mock import mock_registry


def test_get_rank_and_get_score():
    c = Candidate(
        id="a",
        retrieval_score=0.5,
        retrieval={"vector_search": 0.9, "vector_search_rank": 2, "lexical_search_rank": 5},
    )
    ctx = EvalContext(candidate=c)
    assert eval_expr_string('retrieval.get_rank("vector_search", 999)', ctx) == 2
    assert eval_expr_string('retrieval.get_rank("missing", 999)', ctx) == 999
    assert eval_expr_string('retrieval.get_score("vector_search", 0.0)', ctx) == pytest.approx(0.9)


def test_rrf_expression():
    c = Candidate(
        id="a",
        retrieval={"vector_search_rank": 1, "lexical_search_rank": 2},
    )
    ctx = EvalContext(candidate=c)
    val = eval_expr_string(
        '1.0 / (60 + retrieval.get_rank("vector_search", 999))'
        ' + 1.0 / (60 + retrieval.get_rank("lexical_search", 999))',
        ctx,
    )
    expected = 1.0 / (60 + 1) + 1.0 / (60 + 2)
    assert val == pytest.approx(expected)


@pytest.mark.asyncio
async def test_hybrid_rrf_sql_and_yaml_parity():
    bags = {
        "vector_search": [("a", 0.9), ("b", 0.8), ("c", 0.1)],
        "lexical_search": [("c", 0.95), ("a", 0.5), ("d", 0.4)],
    }
    reg = mock_registry(bags)
    sql = """
    SELECT score(
      expression='1.0 / (60 + retrieval.get_rank("vector_search", 999)) + 1.0 / (60 + retrieval.get_rank("lexical_search", 999))'
    ) AS fused
    FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_user(input_user_id=$u),
        name='vector_search',
        limit=10
      ),
      text_search(query='q', mode='lexical', name='lexical_search', limit=10)
    )
    ORDER BY fused
    LIMIT 10
    """
    yaml_cfg = {
        "type": "rank",
        "from": "item",
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "als",
                "query_encoder": {"type": "precomputed_user", "input_user_id": "$u"},
                "name": "vector_search",
                "limit": 10,
            },
            {
                "type": "text_search",
                "input_text_query": "q",
                "mode": {"type": "lexical"},
                "name": "lexical_search",
                "limit": 10,
            },
        ],
        "score": {
            "type": "score_ensemble",
            "value_model": (
                '1.0 / (60 + retrieval.get_rank("vector_search", 999))'
                ' + 1.0 / (60 + retrieval.get_rank("lexical_search", 999))'
            ),
            "output_alias": "fused",
            "preserve_order": False,
        },
        "limit": 10,
    }
    from recql.yaml_query import query_input_to_rank_query_config

    page_sql = await execute(bind(query_input_to_rank_query_config(sql), params={"u": "1"}), reg)
    page_yaml = await execute(
        bind(convert_rank_query_config(yaml_cfg), params={"u": "1"}), reg
    )
    assert page_sql.ids() == page_yaml.ids()
    # a appears in both bags → best RRF; c also in both
    assert page_sql.ids()[0] in ("a", "c")
    # §2.2: default retrieval_score still first-bag for a
    by_id = {c.id: c for c in page_sql.items}
    assert by_id["a"].retrieval["vector_search"] == 0.9
    assert by_id["a"].retrieval["lexical_search"] == 0.5
    assert by_id["a"].retrieval_score == 0.9
