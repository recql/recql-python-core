"""Physical bindings from engine YAML ``data`` / ``index`` → SQL targets.

Drop-in contract: users point RecQL at existing tables (or ``type: query``
views that expose ``item_id`` / ``user_id`` / ``label`` / ``created_at``).
Hardcoded fixture DDL is only the default when catalog omits ``data``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from recql.catalog import EmbeddingSpec, EngineCatalog, LexicalSearchSpec


@dataclass
class EntityTableBinding:
    """How to read one entity plane (item / user / interaction)."""

    role: str  # item | user | interaction
    # Relation fragment usable after FROM — table name or (subquery) AS alias
    from_sql: str
    id_column: str
    # When set, attributes live in a JSONB column (fixture schema).
    attrs_json_column: str | None = None
    # Wide-table mode: hydrate these columns into Candidate.attributes.
    attribute_columns: list[str] | None = None
    created_at_column: str | None = None
    popular_rank_column: str | None = None
    search_tsv_column: str | None = None
    # interaction extras
    user_id_column: str | None = None
    item_id_column: str | None = None
    label_column: str | None = None
    schema_override: dict[str, Any] = field(default_factory=dict)
    queries: dict[str, str] = field(default_factory=dict)
    backend: str | None = None

    def qualify(self, column: str) -> str:
        return f"{column}"


@dataclass
class EmbeddingStoreBinding:
    """Physical store for one entity plane (user or item) of an embedding."""

    from_sql: str = "embeddings"
    name_column: str | None = "embedding_name"
    entity_type_column: str | None = "entity_type"
    entity_id_column: str = "entity_id"
    vector_column: str = "embedding"
    queries: dict[str, str] = field(default_factory=dict)
    backend: str | None = None


@dataclass
class EmbeddingStoreGroup:
    """Embedding vectors split by entity plane (typical for CF / ALS)."""

    user: EmbeddingStoreBinding | None = None
    item: EmbeddingStoreBinding | None = None
    backend: str | None = None

    def binding_for(self, entity_type: str) -> EmbeddingStoreBinding:
        et = str(entity_type or "item").lower()
        if et == "user" and self.user is not None:
            return self.user
        if et == "item" and self.item is not None:
            return self.item
        if self.item is not None and self.user is None:
            return self.item
        if self.user is not None and self.item is None:
            return self.user
        raise KeyError(f"no embedding store binding for entity_type={entity_type!r}")


@dataclass
class ModelStoreBinding:
    from_sql: str = "models"
    name_column: str = "name"
    blob_column: str = "blob"
    feature_spec_column: str = "feature_spec"
    backend: str | None = None


@dataclass
class PaginationKvBinding:
    """Physical store for pagination seen-item exclusion (plugin-configured)."""

    from_sql: str = "pagination_seen"
    key_column: str = "key"
    item_id_column: str = "item_id"
    expires_at_column: str = "expires_at"
    # When True, plugin may CREATE TABLE IF NOT EXISTS on first use.
    ensure_table: bool = True
    backend: str | None = None


@dataclass
class PersonalFilterBinding:
    """Named ``personal_filter`` from ``data.filters`` (exclude-seen style).

    ``user_id_column`` / ``item_id_column`` tell the engine which side to bind:
    ``input_user_id`` → ``WHERE user_id_column = ?`` and ban ``item_id_column``;
    ``input_item_id`` → ``WHERE item_id_column = ?`` and ban ``user_id_column``.
    """

    name: str
    from_sql: str
    user_id_column: str = "user_id"
    item_id_column: str = "item_id"
    filter_type: str = "personal_filter"
    queries: dict[str, str] = field(default_factory=dict)
    backend: str | None = None


@dataclass
class DataBindings:
    """Resolved physical map for one engine."""

    items: EntityTableBinding
    users: EntityTableBinding
    interactions: EntityTableBinding
    embeddings: EmbeddingStoreBinding = field(default_factory=EmbeddingStoreBinding)
    embedding_stores: dict[str, EmbeddingStoreGroup] = field(default_factory=dict)
    embedding_store_map: dict[str, str] = field(default_factory=dict)
    models: ModelStoreBinding = field(default_factory=ModelStoreBinding)
    pagination_kv: PaginationKvBinding = field(default_factory=PaginationKvBinding)
    filters: dict[str, PersonalFilterBinding] = field(default_factory=dict)
    lexical: Any | None = None  # LexicalSearchSpec | None
    embedding_specs: dict[str, Any] = field(default_factory=dict)
    query_templates: dict[str, str] = field(default_factory=dict)
    backend: str = ""

    def entity(self, entity_type: str) -> EntityTableBinding:
        if entity_type == "user":
            return self.users
        return self.items

    def personal_filter(self, name: str) -> PersonalFilterBinding | None:
        key = _normalize_filter_ref(name)
        if key in self.filters:
            return self.filters[key]
        return None

    def embedding_store_for(
        self, embedding_name: str, entity_type: str = "item"
    ) -> EmbeddingStoreBinding:
        if embedding_name in self.embedding_stores:
            return self.embedding_stores[embedding_name].binding_for(entity_type)
        key = self.embedding_store_map.get(embedding_name)
        if key and key in self.embedding_stores:
            return self.embedding_stores[key].binding_for(entity_type)
        return self.embeddings

    def backend_for_embedding(
        self, embedding_name: str, entity_type: str = "item"
    ) -> str | None:
        if embedding_name in self.embedding_stores:
            group = self.embedding_stores[embedding_name]
            binding = group.binding_for(entity_type)
            if binding and binding.backend:
                return binding.backend
            if group.backend:
                return group.backend
        key = self.embedding_store_map.get(embedding_name)
        if key and key in self.embedding_stores:
            group = self.embedding_stores[key]
            binding = group.binding_for(entity_type)
            if binding and binding.backend:
                return binding.backend
            if group.backend:
                return group.backend
        spec = self.embedding_specs.get(embedding_name)
        if hasattr(spec, "backend") and spec.backend:
            return spec.backend
        if isinstance(spec, dict) and spec.get("backend"):
            return str(spec["backend"])
        return self.backend or None

    def backend_for_lexical(self) -> str | None:
        if self.lexical and hasattr(self.lexical, "backend") and self.lexical.backend:
            return self.lexical.backend
        return self.backend or None


def _normalize_filter_ref(ref: str) -> str:
    """``ref:data.filters:exclude_seen`` / ``exclude_seen`` → ``exclude_seen``."""
    raw = str(ref or "").strip()
    if raw.lower().startswith("ref:data.filters:"):
        return raw.split(":", 2)[-1].strip()
    if raw.lower().startswith("data.filters:"):
        return raw.split(":", 1)[-1].strip()
    return raw


def _parse_queries_block(raw: Any) -> dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v).strip() for k, v in raw.items() if v}


def _embedding_store_from_config(cfg: dict[str, Any]) -> EmbeddingStoreBinding:
    name_col = cfg.get("name_column")
    entity_col = cfg.get("entity_type_column")
    return EmbeddingStoreBinding(
        from_sql=_table_or_query_from_config(cfg, default="embeddings"),
        name_column=str(name_col) if name_col is not None else None,
        entity_type_column=str(entity_col) if entity_col is not None else None,
        entity_id_column=str(cfg.get("entity_id_column") or "entity_id"),
        vector_column=str(cfg.get("vector_column") or "embedding"),
        queries=_parse_queries_block(cfg.get("queries")),
        backend=str(cfg["backend"]) if cfg.get("backend") else None,
    )


def _embedding_store_group_from_config(cfg: dict[str, Any]) -> EmbeddingStoreGroup:
    """Parse ``user`` / ``item`` planes, or a legacy unified table definition."""
    backend = str(cfg["backend"]) if cfg.get("backend") else None
    if "user" in cfg or "item" in cfg:
        user_cfg = cfg.get("user")
        item_cfg = cfg.get("item")
        user_b = _embedding_store_from_config(user_cfg) if isinstance(user_cfg, dict) else None
        if user_b and not user_b.backend and backend:
            user_b.backend = backend
        item_b = _embedding_store_from_config(item_cfg) if isinstance(item_cfg, dict) else None
        if item_b and not item_b.backend and backend:
            item_b.backend = backend
        return EmbeddingStoreGroup(
            user=user_b,
            item=item_b,
            backend=backend,
        )
    unified = _embedding_store_from_config(cfg)
    return EmbeddingStoreGroup(user=unified, item=unified, backend=backend)


def _table_or_query_from_config(cfg: dict[str, Any], *, default: str) -> str:
    t = cfg.get("type") or ("query" if "query" in cfg else "table")
    if t == "query" or cfg.get("query"):
        q = str(cfg["query"]).strip().rstrip(";")
        # No leading underscore — Oracle unquoted identifiers must start with a letter.
        return f"({q}) recql_emb"
    return _quote_ident(str(cfg.get("table") or cfg.get("name") or default))


def embedding_store_filters(
    store: EmbeddingStoreBinding,
    *,
    embedding_name: str | None,
    entity_type: str | None,
) -> tuple[str, dict[str, Any]]:
    """SQL ``AND`` clauses + named bind map for an embedding lookup."""
    parts: list[str] = []
    binds: dict[str, Any] = {}
    if store.name_column and embedding_name is not None:
        parts.append(f"emb.{store.name_column} = $embedding_name")
        binds["embedding_name"] = embedding_name
    if store.entity_type_column and entity_type is not None:
        parts.append(f"emb.{store.entity_type_column} = $entity_type")
        binds["entity_type"] = entity_type
    if not parts:
        return "TRUE", {}
    return " AND ".join(parts), binds


def default_fixture_bindings(*, backend: str) -> DataBindings:
    """Convenience bindings matching ``examples/generator`` demo DDL.

    ``backend`` is required (no default). Online RecQL does **not** require this
    schema — point ``data.*_table`` / ``index.embedding_store`` at existing
    tables via engine YAML instead.
    """
    from recql.errors import ExecuteError
    from recql.plugins.dialect import canonical_backend_name, load_default_queries

    if not (backend or "").strip():
        raise ExecuteError("default_fixture_bindings requires backend=")
    backend = canonical_backend_name(backend)

    text_item = EmbeddingStoreBinding(
        from_sql="text_embeddings",
        name_column="embedding_name",
        entity_type_column=None,
        entity_id_column="entity_id",
    )
    als_user = EmbeddingStoreBinding(
        from_sql="als_user_embeddings",
        name_column=None,
        entity_type_column=None,
        entity_id_column="user_id",
    )
    als_item = EmbeddingStoreBinding(
        from_sql="als_item_embeddings",
        name_column=None,
        entity_type_column=None,
        entity_id_column="item_id",
    )
    return DataBindings(
        items=EntityTableBinding(
            role="item",
            from_sql="items",
            id_column="item_id",
            attrs_json_column="attrs",
            created_at_column="created_at",
            popular_rank_column="_derived_popular_rank",
            search_tsv_column="search_tsv",
        ),
        users=EntityTableBinding(
            role="user",
            from_sql="users",
            id_column="user_id",
            attrs_json_column="attrs",
        ),
        interactions=EntityTableBinding(
            role="interaction",
            from_sql="interactions",
            id_column="item_id",
            user_id_column="user_id",
            item_id_column="item_id",
            label_column="label",
            created_at_column="created_at",
        ),
        embeddings=text_item,
        embedding_stores={
            "content_embedding": EmbeddingStoreGroup(item=text_item),
            "title_embedding": EmbeddingStoreGroup(item=text_item),
            "als": EmbeddingStoreGroup(user=als_user, item=als_item),
            # Shared aliases for ``store: text|als`` in engine YAML
            "text": EmbeddingStoreGroup(item=text_item),
        },
        embedding_store_map={
            "content_embedding": "content_embedding",
            "title_embedding": "title_embedding",
            "als": "als",
        },
        filters={
            "exclude_seen": PersonalFilterBinding(
                name="exclude_seen",
                from_sql="interactions",
                user_id_column="user_id",
                item_id_column="item_id",
            ),
        },
        query_templates=load_default_queries(backend),
        backend=backend,
    )


def _quote_ident(name: str) -> str:
    # Allow schema.table
    parts = name.split(".")
    return ".".join(f'"{p}"' if not p.isidentifier() else p for p in parts)


def _table_from_config(cfg: dict[str, Any] | None, *, default_name: str) -> tuple[str, dict[str, Any]]:
    """Return (from_sql, meta) from data.*_table block."""
    if not cfg:
        return default_name, {}
    t = cfg.get("type") or ("query" if "query" in cfg else "table")
    if t == "query" or "query" in cfg:
        q = str(cfg["query"]).strip().rstrip(";")
        # No AS (ORA-03048); no leading underscore (ORA-00911).
        return f"({q}) recql_src", cfg
    name = cfg.get("name") or default_name
    return _quote_ident(str(name)), cfg


def bindings_from_catalog(catalog: Any | None) -> DataBindings:
    """Build DataBindings from engine YAML; fall back to fixture defaults."""
    from recql.catalog.query_templates import query_templates_from_catalog
    from recql.errors import ExecuteError
    from recql.plugins.dialect import canonical_backend_name

    if catalog is None:
        raise ExecuteError(
            "bindings_from_catalog requires a catalog with plugins.backend "
            "(no default backend)"
        )

    plugins = getattr(catalog, "plugins", None) or {}
    raw_backend = plugins.get("backend")
    if not raw_backend:
        dep = getattr(catalog, "deployment", None) or {}
        raw_backend = dep.get("backend")
    if not raw_backend:
        backends = getattr(catalog, "backends", {}) or {}
        if backends:
            raw_backend = next(iter(backends.values())).backend
    if not raw_backend:
        raise ExecuteError(
            "plugins.backend is required in engine YAML (no default backend)"
        )
    backend = canonical_backend_name(str(raw_backend))

    data = catalog.data or {}
    schema = data.get("schema_override") or {}

    item_from, item_cfg = _table_from_config(
        data.get("item_table") or data.get("items") or data.get("item"),
        default_name="items",
    )
    user_from, user_cfg = _table_from_config(
        data.get("user_table") or data.get("users") or data.get("user"),
        default_name="users",
    )
    inter_from, inter_cfg = _table_from_config(
        data.get("interaction_table") or data.get("interactions") or data.get("interaction"),
        default_name="interactions",
    )

    # Detect fixture-shaped vs wide tables via schema_override / explicit columns
    item_cols = _feature_columns(schema.get("item") or schema.get("items"))
    user_cols = _feature_columns(schema.get("user") or schema.get("users"))

    items = EntityTableBinding(
        role="item",
        from_sql=item_from,
        id_column=str(item_cfg.get("id_column") or "item_id"),
        attrs_json_column=(
            str(item_cfg["attrs_column"])
            if item_cfg.get("attrs_column")
            else (None if item_cols else _maybe_attrs_col(item_cfg, default="attrs"))
        ),
        attribute_columns=item_cols,
        created_at_column=str(
            item_cfg.get("created_at_column") or "created_at"
        ),
        popular_rank_column=str(
            item_cfg.get("popular_rank_column") or "_derived_popular_rank"
        ),
        search_tsv_column=(
            str(item_cfg["search_tsv_column"])
            if item_cfg.get("search_tsv_column")
            else ("search_tsv" if not item_cols else None)
        ),
        schema_override=dict(schema.get("item") or {}),
        queries=_parse_queries_block(item_cfg.get("queries")),
        backend=str(item_cfg["backend"]) if item_cfg.get("backend") else None,
    )
    # Optional Oracle-style search text column (not tsvector)
    if item_cfg.get("search_text_column"):
        items.schema_override["search_text_column"] = item_cfg["search_text_column"]

    if item_from == "items" and not item_cols and data.get("item_table") is None:
        items.attrs_json_column = "attrs"
        items.search_tsv_column = "search_tsv"

    users = EntityTableBinding(
        role="user",
        from_sql=user_from,
        id_column=str(user_cfg.get("id_column") or "user_id"),
        attrs_json_column=(
            str(user_cfg["attrs_column"])
            if user_cfg.get("attrs_column")
            else (None if user_cols else _maybe_attrs_col(user_cfg, default="attrs"))
        ),
        attribute_columns=user_cols,
        schema_override=dict(schema.get("user") or {}),
        queries=_parse_queries_block(user_cfg.get("queries")),
        backend=str(user_cfg["backend"]) if user_cfg.get("backend") else None,
    )
    if user_from == "users" and not user_cols and data.get("user_table") is None:
        users.attrs_json_column = "attrs"

    interactions = EntityTableBinding(
        role="interaction",
        from_sql=inter_from,
        id_column=str(inter_cfg.get("item_id_column") or "item_id"),
        user_id_column=str(inter_cfg.get("user_id_column") or "user_id"),
        item_id_column=str(inter_cfg.get("item_id_column") or "item_id"),
        label_column=str(inter_cfg.get("label_column") or "label"),
        created_at_column=str(inter_cfg.get("created_at_column") or "created_at"),
        schema_override=dict(schema.get("interaction") or {}),
        queries=_parse_queries_block(inter_cfg.get("queries")),
        backend=str(inter_cfg["backend"]) if inter_cfg.get("backend") else None,
    )

    emb_store = EmbeddingStoreBinding()
    embedding_stores: dict[str, EmbeddingStoreGroup] = {}
    embedding_store_map: dict[str, str] = {}
    index = (catalog.raw or {}).get("index") or {}
    store = index.get("embedding_store") or data.get("embedding_store")
    if isinstance(store, dict):
        emb_store = _embedding_store_from_config(store)

    stores_cfg = index.get("embedding_stores") or {}
    if isinstance(stores_cfg, dict):
        for key, cfg in stores_cfg.items():
            if isinstance(cfg, dict):
                embedding_stores[str(key)] = _embedding_store_group_from_config(cfg)

    for emb_name, spec in (catalog.embeddings or {}).items():
        raw = getattr(spec, "raw", None) or {}
        inline = raw.get("stores")
        if isinstance(inline, dict):
            embedding_stores[str(emb_name)] = _embedding_store_group_from_config(inline)
        store_key = raw.get("store")
        if store_key:
            embedding_store_map[str(emb_name)] = str(store_key)
        elif str(emb_name) not in embedding_store_map and str(emb_name) in embedding_stores:
            embedding_store_map[str(emb_name)] = str(emb_name)
        elif str(emb_name) == "als" and "als" in embedding_stores:
            embedding_store_map[str(emb_name)] = "als"
        elif str(emb_name) in ("content_embedding", "title_embedding"):
            if str(emb_name) in embedding_stores:
                embedding_store_map[str(emb_name)] = str(emb_name)
            elif "text" in embedding_stores:
                embedding_store_map[str(emb_name)] = "text"

    model_store = ModelStoreBinding()
    mstore = index.get("model_store") or data.get("model_store")
    if isinstance(mstore, dict):
        model_store = ModelStoreBinding(
            from_sql=_quote_ident(str(mstore.get("table") or mstore.get("name") or "models")),
            name_column=str(mstore.get("name_column") or "name"),
            blob_column=str(mstore.get("blob_column") or "blob"),
            feature_spec_column=str(mstore.get("feature_spec_column") or "feature_spec"),
            backend=str(mstore["backend"]) if mstore.get("backend") else None,
        )

    pagination_kv = _pagination_kv_binding(catalog, data)
    filters = _filters_from_config(data, interactions=interactions)

    return DataBindings(
        items=items,
        users=users,
        interactions=interactions,
        embeddings=emb_store,
        embedding_stores=embedding_stores,
        embedding_store_map=embedding_store_map,
        models=model_store,
        pagination_kv=pagination_kv,
        filters=filters,
        lexical=catalog.lexical_search,
        embedding_specs=dict(catalog.embeddings),
        query_templates=query_templates_from_catalog(catalog, backend=backend),
        backend=backend,
    )


def _pagination_kv_binding(catalog: Any, data: dict[str, Any]) -> PaginationKvBinding:
    """Resolve from ``data.pagination_kv`` or ``deployment.pagination.store``."""
    cfg: dict[str, Any] = {}
    dep = getattr(catalog, "deployment", None) or {}
    if isinstance(dep, dict):
        pag = dep.get("pagination") or {}
        if isinstance(pag, dict) and isinstance(pag.get("store"), dict):
            cfg.update(pag["store"])
    raw_data = data.get("pagination_kv") or data.get("pagination_seen")
    if isinstance(raw_data, dict):
        cfg.update(raw_data)
    # Plugin-local override: plugins.<backend>.pagination_kv
    plugins = getattr(catalog, "plugins", None) or {}
    if isinstance(plugins, dict):
        backend = str(plugins.get("backend") or dep.get("backend") or "")
        block = plugins.get(backend) if backend else None
        if isinstance(block, dict) and isinstance(block.get("pagination_kv"), dict):
            cfg.update(block["pagination_kv"])
    if not cfg:
        return PaginationKvBinding()
    ensure = cfg.get("ensure_table", cfg.get("ensure", True))
    return PaginationKvBinding(
        from_sql=_quote_ident(str(cfg.get("table") or cfg.get("name") or "pagination_seen")),
        key_column=str(cfg.get("key_column") or "key"),
        item_id_column=str(cfg.get("item_id_column") or "item_id"),
        expires_at_column=str(cfg.get("expires_at_column") or "expires_at"),
        ensure_table=bool(ensure),
    )


def _filters_from_config(
    data: dict[str, Any], *, interactions: EntityTableBinding
) -> dict[str, PersonalFilterBinding]:
    """Parse ``data.filters``; default ``exclude_seen`` from interaction table."""
    out: dict[str, PersonalFilterBinding] = {}
    raw = data.get("filters")
    entries: list[dict[str, Any]] = []
    if isinstance(raw, list):
        entries = [e for e in raw if isinstance(e, dict)]
    elif isinstance(raw, dict):
        for name, cfg in raw.items():
            if isinstance(cfg, dict):
                entries.append({"name": name, **cfg})
    for entry in entries:
        filt = _personal_filter_from_entry(entry)
        if filt is not None:
            out[filt.name] = filt
    if "exclude_seen" not in out:
        out["exclude_seen"] = PersonalFilterBinding(
            name="exclude_seen",
            from_sql=interactions.from_sql,
            user_id_column=str(interactions.user_id_column or "user_id"),
            item_id_column=str(interactions.item_id_column or "item_id"),
            queries=dict(interactions.queries or {}),
        )
    return out


def _personal_filter_from_entry(entry: dict[str, Any]) -> PersonalFilterBinding | None:
    name = str(entry.get("name") or "").strip()
    if not name:
        return None
    ftype_raw = entry.get("filter_type") or entry.get("type") or "personal_filter"
    if isinstance(ftype_raw, dict):
        ftype = str(ftype_raw.get("type") or "personal_filter")
        user_col = str(
            ftype_raw.get("user_id_column")
            or entry.get("user_id_column")
            or "user_id"
        )
        item_col = str(
            ftype_raw.get("item_id_column")
            or entry.get("item_id_column")
            or "item_id"
        )
    else:
        ftype = str(ftype_raw)
        user_col = str(entry.get("user_id_column") or "user_id")
        item_col = str(entry.get("item_id_column") or "item_id")
    if ftype not in ("personal_filter", "exclude_seen", "personal"):
        # Unknown filter types are ignored for now (expression filters etc.).
        return None

    table_cfg = entry.get("filter_table")
    if table_cfg is None and isinstance(entry.get("table"), dict):
        table_cfg = entry.get("table")
    if not isinstance(table_cfg, dict):
        # Flat shorthand: query / table name on the filter entry itself.
        table_cfg = {}
        if entry.get("query"):
            table_cfg = {"type": "query", "query": entry["query"]}
        elif entry.get("table") and not isinstance(entry.get("table"), dict):
            table_cfg = {"type": "table", "name": entry["table"]}
        elif entry.get("from"):
            table_cfg = {"type": "table", "name": entry["from"]}
        else:
            table_cfg = {"type": "table", "name": "interactions"}

    from_sql, _meta = _table_from_config(table_cfg, default_name="interactions")
    queries = _parse_queries_block(entry.get("queries"))
    queries = {**_parse_queries_block(table_cfg.get("queries")), **queries}
    return PersonalFilterBinding(
        name=name,
        from_sql=from_sql,
        user_id_column=user_col,
        item_id_column=item_col,
        filter_type="personal_filter",
        queries=queries,
        backend=str(entry["backend"]) if entry.get("backend") else None,
    )


def _feature_columns(entity_schema: Any) -> list[str] | None:
    if not isinstance(entity_schema, dict):
        return None
    feats = entity_schema.get("features") or entity_schema.get("columns")
    if isinstance(feats, list) and feats:
        out = []
        for f in feats:
            if isinstance(f, str):
                out.append(f)
            elif isinstance(f, dict) and "name" in f:
                out.append(str(f["name"]))
        return out or None
    if isinstance(feats, dict) and feats:
        return list(feats.keys())
    return None


def _maybe_attrs_col(cfg: dict[str, Any], *, default: str | None) -> str | None:
    if cfg.get("attrs_column"):
        return str(cfg["attrs_column"])
    return default
