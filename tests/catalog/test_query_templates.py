"""Query template rendering — core unit tests (inline catalogs only).

Default-query / named-bind checks exercise *whatever dialects are installed*
via entry points — shapes come from ``SqlDialect.placeholder``, not hardcoded
backend names in core.
"""

from __future__ import annotations

import pytest

from recql.catalog import load_engine_catalog
from recql.catalog.bindings import bindings_from_catalog
from recql.catalog.query_templates import QueryRenderer, load_default_queries, render_query_template
from recql.plugins.dialect import get_sql_dialect

from tests._dialects import installed_dialect_names, require_any_dialect


@pytest.mark.parametrize("backend", installed_dialect_names() or ["__none__"])
def test_default_templates_load(backend: str):
    if backend == "__none__":
        pytest.skip("no recql.dialects packs installed")
    tpl = load_default_queries(backend)
    assert "embedding_vector_search" in tpl
    assert "{emb_from}" in tpl["embedding_vector_search"]
    assert "$query_vector" in tpl["embedding_vector_search"]


@pytest.mark.parametrize("backend", installed_dialect_names() or ["__none__"])
def test_render_named_binds_use_dialect_placeholder(backend: str):
    if backend == "__none__":
        pytest.skip("no recql.dialects packs installed")
    dialect = get_sql_dialect(backend)
    sql, args = render_query_template(
        "SELECT 1 WHERE x = $limit AND y = $query_text",
        structural={},
        binds={"limit": 5, "query_text": "hello"},
        backend=backend,
    )
    assert dialect.placeholder(1) in sql
    assert dialect.placeholder(2) in sql
    assert args == [5, "hello"]


def test_engine_yaml_overrides_plugin_queries():
    backend = require_any_dialect()
    cat = load_engine_catalog(
        {
            "name": "custom",
            "plugins": {
                "backend": backend,
                backend: {
                    "queries": {
                        "entity_filter": "SELECT {item_select} FROM {item_from} e LIMIT $limit",
                    }
                },
            },
            "data": {"item_table": {"name": "catalog_items", "type": "table"}},
        }
    )
    b = bindings_from_catalog(cat)
    renderer = QueryRenderer(b)
    sql, args = renderer.render(
        "entity_filter",
        structural={**renderer.entity_structural(b.items), "where": "TRUE"},
        binds={"limit": 3},
        entity=b.items,
    )
    assert "catalog_items" in sql
    assert args == [3]


def test_embedding_store_split_user_item():
    backend = require_any_dialect()
    cat = load_engine_catalog(
        {
            "name": "legacy",
            "plugins": {"backend": backend},
            "index": {
                "embedding_stores": {
                    "als": {
                        "user": {
                            "table": "user_factors",
                            "entity_id_column": "user_id",
                            "vector_column": "embedding",
                        },
                        "item": {
                            "table": "item_factors",
                            "entity_id_column": "item_id",
                            "vector_column": "embedding",
                        },
                    }
                },
                "embeddings": [{"name": "als", "dims": 32}],
            },
        }
    )
    b = bindings_from_catalog(cat)
    user_store = b.embedding_store_for("als", "user")
    item_store = b.embedding_store_for("als", "item")
    assert user_store.from_sql == "user_factors"
    assert user_store.entity_id_column == "user_id"
    assert item_store.from_sql == "item_factors"
    assert item_store.entity_id_column == "item_id"


def test_embedding_store_query_override():
    backend = require_any_dialect()
    cat = load_engine_catalog(
        {
            "name": "legacy",
            "plugins": {"backend": backend},
            "index": {
                "embedding_stores": {
                    "als": {
                        "user": {
                            "type": "query",
                            "query": "SELECT user_id, factors AS embedding FROM user_factors",
                            "entity_id_column": "user_id",
                            "queries": {
                                "embedding_lookup": (
                                    "SELECT embedding FROM {emb_from} "
                                    "WHERE user_id = $entity_id"
                                ),
                            },
                        }
                    }
                },
                "embeddings": [{"name": "als", "dims": 32}],
            },
        }
    )
    b = bindings_from_catalog(cat)
    store = b.embedding_store_for("als", "user")
    assert "user_factors" in store.from_sql
    renderer = QueryRenderer(b)
    sql, args = renderer.render(
        "embedding_lookup",
        structural={
            "emb_from": store.from_sql,
            "entity_id_column": "user_id",
            "emb_filter": "TRUE",
        },
        binds={"entity_id": "55"},
        store=store,
    )
    assert "user_factors" in sql
    assert args == ["55"]
