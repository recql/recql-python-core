"""Catalog plugin backend + binding column overrides."""

from __future__ import annotations

from recql.catalog import load_engine_catalog
from recql.plugins.dialect import get_sql_dialect
from recql.plugins.factory import plugin_backend_name, plugin_config

from tests._dialects import require_any_dialect


def test_plugins_backend_from_engine_yaml():
    backend = require_any_dialect()
    dialect = get_sql_dialect(backend)
    aliases = [str(a).lower() for a in (getattr(dialect, "aliases", ()) or ())]
    # Prefer an alias when the pack registers one — exercises canonical resolution.
    configured = aliases[0] if aliases else backend

    cat = load_engine_catalog(
        {
            "name": "t",
            "plugins": {
                "backend": configured,
                backend: {"vector_distance": "COSINE"},
            },
            "data": {
                "item_table": {
                    "name": "catalog_items",
                    "type": "table",
                    "id_column": "sku",
                    "popular_rank_column": "pop_rank",
                }
            },
            "deployment": {
                "pagination": {
                    "store": {
                        "table": "recql_seen",
                        "key_column": "session_key",
                        "item_id_column": "sku",
                        "expires_at_column": "exp_ts",
                        "ensure_table": False,
                    }
                }
            },
        }
    )
    assert plugin_backend_name(cat) == backend
    assert plugin_config(cat)["vector_distance"] == "COSINE"
    b = cat.bindings()
    assert b.items.from_sql == "catalog_items"
    assert b.items.id_column == "sku"
    assert b.items.popular_rank_column == "pop_rank"
    kv = b.pagination_kv
    assert kv.from_sql == "recql_seen"
    assert kv.key_column == "session_key"
    assert kv.item_id_column == "sku"
    assert kv.expires_at_column == "exp_ts"
    assert kv.ensure_table is False
