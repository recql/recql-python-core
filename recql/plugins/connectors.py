"""Discover installed RecQL backend connectors (separate installable packs)."""

from __future__ import annotations

from importlib.metadata import entry_points
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
