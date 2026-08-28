"""Catalog DataBindings tests."""

from __future__ import annotations

import pytest

from recql.catalog import load_engine_catalog
from recql.catalog.bindings import bindings_from_catalog, default_fixture_bindings
from recql.errors import ExecuteError

from tests._dialects import require_any_dialect


def test_default_fixture_bindings():
    backend = require_any_dialect()
    b = default_fixture_bindings(backend=backend)
    assert b.items.from_sql == "items"
    assert b.items.id_column == "item_id"
    assert b.items.attrs_json_column == "attrs"
    assert b.embeddings.from_sql == "text_embeddings"
    assert b.backend == backend


def test_default_fixture_bindings_requires_backend():
    with pytest.raises(ExecuteError, match="requires backend"):
        default_fixture_bindings(backend="")


def test_bindings_from_named_tables():
    backend = require_any_dialect()
    cat = load_engine_catalog(
        {
            "name": "drop_in",
            "plugins": {"backend": backend},
            "data": {
                "item_table": {"name": "apparel_catalog", "type": "table"},
                "user_table": {"name": "customers", "type": "table"},
                "interaction_table": {"name": "transactions", "type": "table"},
                "schema_override": {
                    "item": {
                        "features": ["color", "name", "department", "created_at"],
                    }
                },
            },
            "index": {
                "embedding_store": {
                    "table": "my_vectors",
                    "name_column": "model",
                    "entity_id_column": "sku",
                },
                "embeddings": [{"name": "content", "encoder": {"type": "hugging_face"}}],
            },
        }
    )
    b = cat.bindings()
    assert b.items.from_sql == "apparel_catalog"
    assert b.items.attrs_json_column is None
    assert b.items.attribute_columns == ["color", "name", "department", "created_at"]
    assert b.users.from_sql == "customers"
    assert b.interactions.from_sql == "transactions"
    assert b.embeddings.from_sql == "my_vectors"
    assert b.embeddings.name_column == "model"
    assert b.embeddings.entity_id_column == "sku"
    assert "exclude_seen" in b.filters
    assert b.filters["exclude_seen"].from_sql == "transactions"


def test_bindings_require_plugins_backend():
    cat = load_engine_catalog(
        {
            "name": "no_backend",
            "data": {"item_table": {"name": "items", "type": "table"}},
        }
    )
    with pytest.raises(ExecuteError, match="plugins.backend"):
        bindings_from_catalog(cat)


def test_bindings_personal_filters_from_config():
    backend = require_any_dialect()
    cat = load_engine_catalog(
        {
            "name": "filters",
            "plugins": {"backend": backend},
            "data": {
                "interaction_table": {"name": "interactions", "type": "table"},
                "filters": [
                    {
                        "name": "exclude_seen_items",
                        "filter_type": {
                            "type": "personal_filter",
                            "user_id_column": "user_id",
                            "item_id_column": "item_id",
                        },
                        "filter_table": {
                            "type": "query",
                            "query": "SELECT item_id, user_id FROM reviews",
                        },
                    }
                ],
            },
        }
    )
    b = bindings_from_catalog(cat)
    filt = b.personal_filter("exclude_seen_items")
    assert filt is not None
    assert filt.user_id_column == "user_id"
    assert filt.item_id_column == "item_id"
    assert "reviews" in filt.from_sql
    assert filt.from_sql.startswith("(")
    assert b.personal_filter("ref:data.filters:exclude_seen_items") is filt
    assert b.personal_filter("exclude_seen") is not None


def test_render_personal_filter_ids_user_and_item():
    from recql.plugins.personal_filter import render_personal_filter_ids

    backend = require_any_dialect()
    cat = load_engine_catalog(
        {
            "name": "pf",
            "plugins": {"backend": backend},
            "data": {
                "filters": [
                    {
                        "name": "exclude_seen",
                        "filter_type": {
                            "type": "personal_filter",
                            "user_id_column": "uid",
                            "item_id_column": "iid",
                        },
                        "filter_table": {"type": "table", "name": "reviews"},
                    }
                ]
            },
        }
    )
    b = bindings_from_catalog(cat)
    filt = b.personal_filter("exclude_seen")
    assert filt is not None
    sql_u, args_u = render_personal_filter_ids(b, filt, entity_id="u1", by_user=True)
    assert "reviews" in sql_u
    assert "iid" in sql_u
    assert "uid" in sql_u
    assert args_u == ["u1"]
    sql_i, args_i = render_personal_filter_ids(b, filt, entity_id="i9", by_user=False)
    assert "uid" in sql_i
    assert args_i == ["i9"]


def test_bindings_from_query_alias():
    backend = require_any_dialect()
    cat = load_engine_catalog(
        {
            "name": "aliased",
            "plugins": {"backend": backend},
            "data": {
                "item_table": {
                    "type": "query",
                    "query": "select product_id as item_id, color, name from apparel_catalog",
                }
            },
        }
    )
    b = bindings_from_catalog(cat)
    assert b.items.from_sql.startswith("(")
    assert "product_id as item_id" in b.items.from_sql
    assert "recql_src" in b.items.from_sql
    assert "AS _recql_src" not in b.items.from_sql
    assert "AS recql_src" not in b.items.from_sql
