"""Engine YAML + plugin dialects → retriever SQL templates.

Named-bind compilation and bundled default SQL live in backend plugins
(``recql.plugins.dialect``). This module only merges/resolves templates and
dispatches driver-specific work to the active dialect.
"""

from __future__ import annotations

from typing import Any

from recql.catalog.bindings import (
    DataBindings,
    EmbeddingStoreBinding,
    EntityTableBinding,
    embedding_store_filters,
)
from recql.errors import ExecuteError
from recql.plugins.dialect import (
    canonical_backend_name,
    compile_named_binds,
    get_sql_dialect,
    load_default_queries,
    normalize_backend_name,
)


def merge_query_templates(*layers: dict[str, str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for layer in layers:
        if layer:
            out.update({str(k): str(v).strip() for k, v in layer.items() if v})
    return out


def _parse_queries_block(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v).strip() for k, v in raw.items() if v}


def query_templates_from_catalog(catalog: Any | None, *, backend: str) -> dict[str, str]:
    """Merge plugin defaults with engine YAML ``plugins.<backend>.queries`` overrides."""
    from recql.plugins.dialect import canonical_backend_name

    if not (backend or "").strip():
        raise ExecuteError("query_templates_from_catalog requires backend=")
    backend = canonical_backend_name(backend)
    layers: list[dict[str, str]] = [load_default_queries(backend)]
    if catalog is None:
        return merge_query_templates(*layers)

    plugins = getattr(catalog, "plugins", None) or {}
    if isinstance(plugins, dict):
        block = plugins.get(backend) or plugins.get(normalize_backend_name(str(plugins.get("backend") or "")))
        if isinstance(block, dict):
            layers.append(_parse_queries_block(block.get("queries")))

    raw = getattr(catalog, "raw", None) or {}
    index = raw.get("index") or {}
    lex = index.get("lexical_search")
    if isinstance(lex, dict):
        layers.append(_parse_queries_block(lex.get("queries")))

    data = getattr(catalog, "data", None) or raw.get("data") or {}
    for key in ("item_table", "user_table", "interaction_table"):
        cfg = data.get(key)
        if isinstance(cfg, dict):
            layers.append(_parse_queries_block(cfg.get("queries")))

    return merge_query_templates(*layers)


def render_query_template(
    template: str,
    *,
    structural: dict[str, Any],
    binds: dict[str, Any],
    backend: str,
) -> tuple[str, list[Any]]:
    """Fill ``{structural}`` fragments then compile ``$binds`` via the plugin dialect."""
    try:
        sql = template.format(**structural)
    except KeyError as exc:
        raise ExecuteError(f"retriever query template missing structural key: {exc}") from exc
    return get_sql_dialect(backend).compile_named_binds(sql, binds)


class QueryRenderer:
    """Resolve and render retriever SQL from defaults + engine overrides."""

    def __init__(self, bindings: DataBindings) -> None:
        self.bindings = bindings
        if not (bindings.backend or "").strip():
            raise ExecuteError("DataBindings.backend is required (no default backend)")
        self.backend = canonical_backend_name(bindings.backend)
        self.dialect = get_sql_dialect(self.backend)

    def resolve(
        self,
        name: str,
        *,
        store: EmbeddingStoreBinding | None = None,
        entity: EntityTableBinding | None = None,
    ) -> str:
        if store is not None and name in store.queries:
            return store.queries[name]
        if entity is not None and name in entity.queries:
            return entity.queries[name]
        if name in self.bindings.query_templates:
            return self.bindings.query_templates[name]
        defaults = self.dialect.default_queries()
        if name not in defaults:
            raise ExecuteError(f"unknown retriever query template: {name}")
        return defaults[name]

    def render(
        self,
        name: str,
        *,
        structural: dict[str, Any],
        binds: dict[str, Any],
        store: EmbeddingStoreBinding | None = None,
        entity: EntityTableBinding | None = None,
    ) -> tuple[str, list[Any]]:
        template = self.resolve(name, store=store, entity=entity)
        try:
            sql = template.format(**structural)
        except KeyError as exc:
            raise ExecuteError(
                f"retriever query template missing structural key: {exc}"
            ) from exc
        return self.dialect.compile_named_binds(sql, binds)

    def entity_structural(self, entity: EntityTableBinding) -> dict[str, str]:
        return {
            "item_from": entity.from_sql,
            "interaction_from": entity.from_sql,
            "item_select": self.dialect.select_entity_columns(entity, alias="e"),
            "id_column": entity.id_column,
            "item_id_column": entity.item_id_column or entity.id_column,
            "user_id_column": entity.user_id_column or "user_id",
            "created_at_column": entity.created_at_column or "created_at",
            "search_tsv_column": entity.search_tsv_column or "search_tsv",
            "search_text_column": str(
                (entity.schema_override or {}).get("search_text_column") or "search_text"
            ),
        }

    def embedding_structural(
        self,
        emb: EmbeddingStoreBinding,
        *,
        embedding_name: str | None,
        entity_type: str | None,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        emb_filter, emb_binds = embedding_store_filters(
            emb,
            embedding_name=embedding_name,
            entity_type=entity_type,
        )
        return (
            {
                "emb_from": emb.from_sql,
                "entity_id_column": emb.entity_id_column,
                "vector_column": emb.vector_column,
                "name_column": emb.name_column or "",
                "entity_type_column": emb.entity_type_column or "",
                "emb_filter": emb_filter,
            },
            emb_binds,
        )


# Re-export for callers/tests that historically imported from this module.
__all__ = [
    "QueryRenderer",
    "compile_named_binds",
    "load_default_queries",
    "merge_query_templates",
    "query_templates_from_catalog",
    "render_query_template",
]
