"""Public harness: detect input → RankQueryConfig → bind → execute."""

from __future__ import annotations

import asyncio
from typing import Any

from recql.bind import BoundRankQuery, bind
from recql.catalog import EngineCatalog, load_engine_catalog
from recql.execute import ResultPage, execute
from recql.language.parser import FeatureFlags, parse
from recql.lower import lower_select, lower_to_dict
from recql.openapi_ir import RankQueryConfig, rank_query_config_to_dict
from recql.plugins.base import PluginRegistry
from recql.plugins.mock import mock_registry
from recql.yaml_query import query_input_to_rank_query_config


def recql_to_rank_query_config(
    query: Any,
    *,
    flags: FeatureFlags | None = None,
) -> dict[str, Any]:
    """Parse/lower or load query → normalized RankQueryConfig dict."""
    if isinstance(query, str) and not _looks_like_yaml(query):
        stmt = parse(query, flags=flags)
        return lower_to_dict(stmt)
    cfg = query_input_to_rank_query_config(query)
    return rank_query_config_to_dict(cfg)


async def recql(
    *,
    engine: EngineCatalog | dict[str, Any] | str | None = None,
    query: Any,
    params: dict[str, Any] | None = None,
    pagination_key: str | None = None,
    backend: PluginRegistry | dict[str, PluginRegistry] | None = None,
    flags: FeatureFlags | None = None,
    timeout_s: float | None = None,
) -> ResultPage:
    """Execute RecQL text or RankQueryConfig / QueryDefinition via single IR path."""
    catalog: EngineCatalog | None
    if engine is None:
        catalog = None
    elif isinstance(engine, EngineCatalog):
        catalog = engine
    else:
        catalog = load_engine_catalog(engine)

    if isinstance(query, str) and not _looks_like_yaml(query):
        stmt = parse(query, flags=flags)
        cfg = lower_select(stmt)
    else:
        cfg = query_input_to_rank_query_config(query)

    bound = bind(cfg, catalog, params=params or {})
    if isinstance(backend, dict):
        from recql.plugins.composite import CompositePluginRegistry

        registry: PluginRegistry = CompositePluginRegistry(registries=backend)
    elif backend is not None:
        registry = backend
    elif catalog is not None and catalog.is_multi_backend():
        from recql.plugins.composite import CompositePluginRegistry

        sub_registries = {bname: mock_registry({}) for bname in catalog.backends}
        registry = CompositePluginRegistry(registries=sub_registries)
    else:
        # In-process mock when no PluginRegistry is supplied (plan / unit paths).
        registry = mock_registry({})
    if catalog is not None and getattr(registry, "_recql_catalog", None) is None:
        try:
            registry._recql_catalog = catalog  # type: ignore[attr-defined]
            registry._recql_bindings = catalog.bindings()  # type: ignore[attr-defined]
        except Exception:
            pass
    return await execute(
        bound, registry, pagination_key=pagination_key, timeout_s=timeout_s
    )


def recql_sync(**kwargs: Any) -> ResultPage:
    """Sync wrapper; forbids nested event-loop usage like graphql execute_sync."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(recql(**kwargs))
    raise RuntimeError("recql_sync() cannot be called from a running event loop; use await recql()")


def _looks_like_yaml(text: str) -> bool:
    from recql.yaml_query import _looks_like_structured_query

    return _looks_like_structured_query(text)
