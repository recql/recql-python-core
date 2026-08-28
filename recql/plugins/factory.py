"""Plugin pack selection from engine YAML ``plugins.backend``.

RecQL core is storage-agnostic. Backend packs register via setuptools entry
points (``recql.backends``); aliases are pack-owned (extra EP names and/or
dialect ``aliases``). Core never hardcodes backend names.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Any, Awaitable, Callable

from recql.catalog import EngineCatalog
from recql.errors import ExecuteError
from recql.plugins.base import PluginRegistry
from recql.plugins.dialect import canonical_backend_name, normalize_backend_name

BackendOpener = Callable[..., Awaitable[PluginRegistry]]


def plugin_backend_name(catalog: EngineCatalog | None) -> str | None:
    """Return the configured backend, or ``None`` if unset.

    There is no default backend — engine YAML (or the caller) must set
    ``plugins.backend`` / ``deployment.backend``.
    """
    if catalog is None:
        return None
    plugins = catalog.plugins or {}
    if plugins.get("backend"):
        return canonical_backend_name(str(plugins["backend"]))
    dep = catalog.deployment or {}
    if dep.get("backend"):
        return canonical_backend_name(str(dep["backend"]))
    return None


def plugin_config(catalog: EngineCatalog | None, backend: str | None = None) -> dict[str, Any]:
    if catalog is None:
        return {}
    raw = backend or plugin_backend_name(catalog)
    if not raw:
        return {}
    canon = canonical_backend_name(raw)
    plugins = catalog.plugins or {}
    block = plugins.get(canon) or plugins.get(normalize_backend_name(raw))
    return dict(block) if isinstance(block, dict) else {}


def _backend_entry_points() -> dict[str, Any]:
    eps = entry_points()
    selected = (
        eps.select(group="recql.backends")
        if hasattr(eps, "select")
        else eps.get("recql.backends", [])
    )
    out: dict[str, Any] = {}
    for ep in selected:
        out[str(ep.name).lower()] = ep
    return out


def resolve_backend_opener(backend: str) -> BackendOpener | None:
    """Return the registered ``open_registry`` callable for ``backend``, if any."""
    key = normalize_backend_name(backend)
    if not key:
        return None
    eps = _backend_entry_points()
    # Exact EP name (canonical or pack-registered alias).
    ep = eps.get(key)
    if ep is None:
        # Dialect aliases → canonical pack name → EP.
        try:
            canon = canonical_backend_name(key)
        except ExecuteError:
            canon = key
        ep = eps.get(canon)
    if ep is None:
        return None
    return ep.load()


async def open_registry(
    *,
    catalog: EngineCatalog | None = None,
    pool: Any = None,
    connection: Any = None,
    backend: str | None = None,
    **kwargs: Any,
) -> PluginRegistry:
    """Build the plugin registry for ``plugins.backend``.

    Installed backend packs register themselves via ``recql.backends``.
    ``mock`` is always available from core when requested explicitly.
    """
    raw = backend if backend is not None else plugin_backend_name(catalog)
    if not raw:
        raise ExecuteError(
            "plugins.backend is required (no default backend). "
            "Set it in engine YAML or pass backend= to open_registry()."
        )
    name = canonical_backend_name(raw)
    cfg = plugin_config(catalog, name)

    if name == "mock":
        from recql.plugins.mock import mock_registry

        return mock_registry(kwargs.get("bags_by_name") or {})

    opener = resolve_backend_opener(name)
    if opener is None:
        # Also try the raw key in case only an alias EP is registered.
        opener = resolve_backend_opener(normalize_backend_name(raw))
    if opener is None:
        installed = sorted(_backend_entry_points())
        raise ExecuteError(
            f"no RecQL backend plugin registered for {raw!r}. "
            f"Install the matching pack (e.g. recql-{name}). "
            f"Registered backends: {installed or ['(none)']}"
        )

    return await opener(
        catalog=catalog,
        pool=pool,
        connection=connection,
        plugin_cfg=cfg,
        **kwargs,
    )
