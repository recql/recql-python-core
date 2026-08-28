"""Expression builtin tests (Phase E)."""

from __future__ import annotations

import math

import pytest

from recql.execute.merge import Candidate
from recql.expr import EvalContext, eval_expr_string


def test_haversine_distance_nyc_approx():
    c = Candidate(id="x")
    ctx = EvalContext(candidate=c)
    # Empire State ↔ Statue of Liberty ~9 km-ish
    d = eval_expr_string(
        "haversine_distance(40.7484, -73.9857, 40.6892, -74.0445)", ctx
    )
    assert 8.0 < float(d) < 12.0


def test_dot_and_cosine():
    c = Candidate(id="x")
    ctx = EvalContext(candidate=c)
    assert eval_expr_string("dot([1,0], [1,0])", ctx) == pytest.approx(1.0)
    assert eval_expr_string("cosine_similarity([1,0], [0,1])", ctx) == pytest.approx(0.0)


def test_now_seconds():
    c = Candidate(id="x")
    ctx = EvalContext(candidate=c)
    v = eval_expr_string("now_seconds()", ctx)
    assert float(v) > 1_700_000_000


def test_text_encoding_from_extras():
    c = Candidate(id="i1")
    ctx = EvalContext(
        candidate=c,
        extras={
            "encodings": {
                "item:text_embedding": [1.0, 0.0],
                "user:text_embedding": [1.0, 0.0],
            }
        },
    )
    v = eval_expr_string(
        "cosine_similarity("
        "text_encoding('item', embedding_ref='text_embedding'), "
        "text_encoding('user', embedding_ref='text_embedding'))",
        ctx,
    )
    assert float(v) == pytest.approx(1.0)


def test_array_and_regexp():
    c = Candidate(id="x", attributes={"tags": ["a", "b"], "title": "Hello World"})
    ctx = EvalContext(candidate=c, item=dict(c.attributes))
    assert eval_expr_string("array_has(tags, 'a')", ctx) is True
    assert eval_expr_string("array_has_any(tags, ['z', 'b'])", ctx) is True
    assert eval_expr_string("array_has_all(tags, ['a', 'b'])", ctx) is True
    assert eval_expr_string("regexp_match(title, '^Hello')", ctx) is True


def test_colbert_stub_and_math():
    c = Candidate(id="x", attributes={"title": "space movie"})
    ctx = EvalContext(candidate=c, item=dict(c.attributes), params={"query": "space"})
    v = eval_expr_string("colbert_v2(item, $query)", ctx)
    assert float(v) > 0.0
    assert eval_expr_string("exp(0)", ctx) == pytest.approx(1.0)
    assert eval_expr_string("max(1, 3, 2)", ctx) == 3
    assert eval_expr_string("round(3.14159, 2)", ctx) == pytest.approx(3.14)

