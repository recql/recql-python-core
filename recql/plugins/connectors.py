"""Discover installed RecQL backend connectors (separate installable packs)."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Awaitable, Callable

from recql.catalog import EngineCatalog
from recql.errors import ExecuteError
from recql.plugins.base import PluginRegistry
from recql.plugins.dialect import canonical_backend_name, normalize_backend_name


Connector = Callable[..., Awaitable[tuple[PluginRegistry, Callable[[], Awaitable[None]]]]]


def _connector_entry_points() -> dict[str, Any]:
    eps = entry_points()
    selected = (
        eps.select(group="recql.connectors")
        if hasattr(eps, "select")
        else eps.get("recql.connectors", [])
    )
    return {str(ep.name).lower(): ep for ep in selected}


def resolve_connector(backend: str) -> Connector | None:
    key = normalize_backend_name(backend)
    if not key:
        return None
    eps = _connector_entry_points()
    ep = eps.get(key)
    if ep is None:
        try:
            canon = canonical_backend_name(key)
        except ExecuteError:
            canon = key
        ep = eps.get(canon)
    if ep is None:
        return None
    return ep.load()


async def open_connection(
    dsn: str,
    *,
    backend: str,
    catalog: EngineCatalog | None = None,
    **kwargs: Any,
) -> tuple[PluginRegistry, Callable[[], Awaitable[None]]]:
    """Open ``(registry, close)`` via the installed ``recql.connectors`` pack for ``backend``."""
    if not normalize_backend_name(backend):
        raise ExecuteError("backend is required (no default backend)")
    name = canonical_backend_name(backend)
    opener = resolve_connector(name) or resolve_connector(normalize_backend_name(backend))
    if opener is None:
        installed = sorted(_connector_entry_points())
        raise ExecuteError(
            f"no RecQL connector registered for {backend!r}. "
            f"Install the backend pack that provides recql.connectors. "
            f"Registered connectors: {installed or ['(none)']}"
        )
    return await opener(dsn, catalog=catalog, **kwargs)


async def open_engine(
    engine_or_catalog: str | Path | dict[str, Any] | EngineCatalog,
    *,
    backend: str | None = None,
    dsn: str | None = None,
    **kwargs: Any,
) -> tuple[PluginRegistry, Callable[[], Awaitable[None]]]:
    """Open a single or multi-backend PluginRegistry and combined close function for an engine catalog."""
    if isinstance(engine_or_catalog, EngineCatalog):
        cat = engine_or_catalog
    else:
        from recql.catalog import load_engine_catalog

        cat = load_engine_catalog(engine_or_catalog)

    if cat.is_multi_backend():
        registries: dict[str, PluginRegistry] = {}
        closers: list[Callable[[], Awaitable[None]]] = []
        for bname, bspec in cat.backends.items():
            b_dsn = bspec.dsn or dsn
            if not b_dsn:
                raise ExecuteError(
                    f"Backend {bname!r} in engine {cat.name!r} has no DSN configured."
                )
            merged_kw = {**bspec.options, **kwargs}
            reg, closer = await open_connection(
                b_dsn,
                backend=bspec.backend,
                catalog=cat,
                **merged_kw,
            )
            registries[bname] = reg
            if bspec.backend not in registries:
                registries[bspec.backend] = reg
            closers.append(closer)

        from recql.plugins.composite import CompositePluginRegistry

        default_b = next(iter(cat.backends.keys()))
        composite_reg = CompositePluginRegistry(
            registries=registries,
            default_backend=default_b,
        )

        async def close_all() -> None:
            for c in closers:
                await c()

        return composite_reg, close_all

    b_name = backend or (
        next(iter(cat.backends.values())).backend if cat.backends else None
    )
    if not b_name:
        from recql.plugins.factory import plugin_backend_name

        b_name = plugin_backend_name(cat)
    if not b_name:
        raise ExecuteError(f"No backend specified for engine {cat.name!r}.")

    b_dsn = dsn or (
        next(iter(cat.backends.values())).dsn if cat.backends else None
    )
    if not b_dsn:
        dep = cat.deployment or {}
        b_dsn = dep.get("dsn")
    if not b_dsn:
        raise ExecuteError(
            f"No DSN specified for backend {b_name!r} in engine {cat.name!r}."
        )

    return await open_connection(b_dsn, backend=b_name, catalog=cat, **kwargs)


__all__ = [
    "Connector",
    "open_connection",
    "open_engine",
    "resolve_connector",
]
