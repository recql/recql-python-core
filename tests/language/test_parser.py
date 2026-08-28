"""Parser golden tests."""

from __future__ import annotations

import pytest

from recql.errors import SyntaxError_, UnsupportedError
from recql.language import parse, parse_expr
from recql.language import ast as A


def test_basic_retrieve_select():
    stmt = parse(
        "SELECT * FROM retrieve(similarity(embedding_ref='als', "
        "encoder=precomputed_user(input_user_id=$user_id), name='uv', limit=50)) LIMIT 10"
    )
    assert len(stmt.select_list) == 1
    assert stmt.select_list[0].star
    assert len(stmt.from_sources) == 1
    src = stmt.from_sources[0]
    assert isinstance(src, A.RetrieveCall)
    assert len(src.calls) == 1
    assert src.calls[0].name.lower() == "similarity"
    assert stmt.limit == 10


def test_where_named_arg_predicate():
    stmt = parse(
        "SELECT * FROM retrieve(filter(where=(category = 'electronics' AND price < 100), limit=20))"
    )
    call = stmt.from_sources[0].calls[0]  # type: ignore[union-attr]
    where_arg = next(a for a in call.args if a.name and a.name.lower() == "where")
    assert isinstance(where_arg.value, A.BinaryOp)
    assert where_arg.value.op == "AND"


def test_bare_in_param():
    expr = parse_expr("category IN $categories")
    assert isinstance(expr, A.InPredicate)
    assert expr.param is not None
    assert expr.param.value == "$categories"


def test_between_and_not_boolean():
    expr = parse_expr("price BETWEEN 10 AND 20 AND active = true")
    assert isinstance(expr, A.BinaryOp)
    assert expr.op == "AND"
    assert isinstance(expr.left, A.BetweenPredicate)


def test_power_right_associative():
    expr = parse_expr("2 ** 3 ** 2")
    assert isinstance(expr, A.BinaryOp)
    assert expr.op == "**"
    assert isinstance(expr.right, A.BinaryOp)
    assert expr.right.op == "**"


def test_multiple_statements_error():
    with pytest.raises(SyntaxError_, match="multiple statements"):
        parse("SELECT * FROM items; SELECT * FROM users")


def test_score_requires_alias():
    with pytest.raises(SyntaxError_, match="alias"):
        parse("SELECT score(expression='x') FROM retrieve(filter(limit=1))")


def test_comment_and_case_insensitive_keywords():
    stmt = parse(
        """
        -- a comment
        select * from RETRIEVE(column_order(columns='created_at DESC', name='new'))
        order by new_score desc nulls last
        limit 5
        """
    )
    assert stmt.limit == 5
    assert stmt.order_by[0].direction == "DESC"
    assert stmt.order_by[0].nulls == "LAST"


def test_with_not_in_language():
    """WITH is not RecQL grammar — rejected at the keyword, not parsed as CTE."""
    with pytest.raises(UnsupportedError, match="WITH clause"):
        parse(
            "WITH x AS (SELECT * FROM retrieve(filter(limit=1))) "
            "SELECT * FROM x LIMIT 1"
        )


def test_from_subquery_not_in_language():
    with pytest.raises(UnsupportedError, match="Subqueries in FROM"):
        parse(
            "SELECT * FROM ("
            "SELECT * FROM retrieve(filter(limit=1))"
            ") LIMIT 1"
        )


def test_join_not_in_language():
    with pytest.raises(UnsupportedError, match="JOIN"):
        parse("SELECT * FROM items JOIN users ON true")
