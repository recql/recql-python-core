"""Dialect contract — relational SQL packs only (``sql_dialect`` feature).

Asserts dialect *shapes* via the registered ``SqlDialect`` for
``recql_testbed.backend``. No hardcoded backend-name branches.
"""

from __future__ import annotations

import pytest

from recql.catalog.bindings import EntityTableBinding
from recql.plugins.dialect import get_sql_dialect, load_default_queries
from recql.testing.features import SQL_DIALECT, require
from recql.testing.testbed import RecqlTestbed


@pytest.mark.asyncio
async def test_named_binds_match_driver(recql_testbed: RecqlTestbed):
    require(recql_testbed, SQL_DIALECT)
    dialect = get_sql_dialect(recql_testbed.backend)
    sql, args = dialect.compile_named_binds(
        "SELECT 1 WHERE id = $entity_id",
        {"entity_id": "x"},
    )
    assert dialect.placeholder(1) in sql
    assert args == ["x"]


@pytest.mark.asyncio
async def test_default_queries_bundled(recql_testbed: RecqlTestbed):
    require(recql_testbed, SQL_DIALECT)
    dialect = get_sql_dialect(recql_testbed.backend)
    assert getattr(dialect, "queries_path").name == "queries.yaml"
    qs = load_default_queries(recql_testbed.backend)
    assert "personal_filter_ids" in qs
    assert "embedding_vector_search" in qs


@pytest.mark.asyncio
async def test_select_entity_columns_attrs(recql_testbed: RecqlTestbed):
    require(recql_testbed, SQL_DIALECT)
    b = EntityTableBinding(
        role="item",
        from_sql="items",
        id_column="item_id",
        attrs_json_column="attrs",
    )
    sql = get_sql_dialect(recql_testbed.backend).select_entity_columns(b, alias="e")
    assert "e.item_id AS entity_id" in sql
    assert "e.attrs AS attrs" in sql


@pytest.mark.asyncio
async def test_wide_table_and_order_by_are_dialect_specific(recql_testbed: RecqlTestbed):
    """Wide SELECT / ORDER BY differ per dialect; assert structural contracts only."""
    require(recql_testbed, SQL_DIALECT)
    dialect = get_sql_dialect(recql_testbed.backend)
    b = EntityTableBinding(
        role="item",
        from_sql="catalog",
        id_column="sku",
        attribute_columns=["color", "name"],
    )
    wide = dialect.select_entity_columns(b)
    assert "sku" in wide or "entity_id" in wide
    assert "color" in wide
    assert "name" in wide

    attrs = EntityTableBinding(
        role="item",
        from_sql="items",
        id_column="item_id",
        attrs_json_column="attrs",
    )
    order = dialect.order_by_sql(attrs, [{"name": "genre", "ascending": True}])
    assert order.strip()
    # Dialect must reference the sort key somehow (column or JSON path).
    assert "genre" in order
