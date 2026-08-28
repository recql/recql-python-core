"""Backend SQL dialect hooks — plugins own driver-specific SQL details.

Shared catalog/query code must not branch on postgres/oracle/mariadb.
Instead it calls ``get_sql_dialect(backend)`` and uses the plugin dialect.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Protocol, TYPE_CHECKING

import yaml

from recql.errors import ExecuteError

if TYPE_CHECKING:
    from recql.catalog.bindings import EntityTableBinding

_NAMED_BIND = re.compile(r"\$(\w+)\b")
_REGISTRY: dict[str, "SqlDialect"] = {}


class SqlDialect(Protocol):
    """Plugin-provided SQL dialect for binds, defaults, and entity SQL fragments."""

    name: str

    def compile_named_binds(
        self, sql: str, binds: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        """Replace ``$name`` placeholders with this driver's bind markers."""
        ...

    def default_queries(self) -> dict[str, str]:
        """Bundled retriever SQL templates for this backend."""
        ...

    def select_entity_columns(
        self, binding: "EntityTableBinding", *, alias: str = "e"
    ) -> str:
        """SELECT list fragment for id + attributes hydration."""
        ...

    def order_by_sql(
        self,
        binding: "EntityTableBinding",
        columns: list[Any],
        *,
        alias: str = "e",
    ) -> str:
        """ORDER BY fragment for column_order retrieve steps."""
        ...

    def render_entity_by_ids(
        self,
        renderer: Any,
        binding: "EntityTableBinding",
        ids: list[str],
    ) -> tuple[str, list[Any]]:
        """Compile ``entity_by_ids`` for this driver's list-bind style."""
        ...


@dataclass
class NamedBindDialect:
    """Helper dialect: shared ``$name`` walk; plugin supplies placeholder style.

    Subclasses must implement ``select_entity_columns`` / ``order_by_sql``.
    """

    name: str
    placeholder: Callable[[int], str]
    queries_path: Path
    aliases: tuple[str, ...] = ()
    _queries: dict[str, str] | None = field(default=None, init=False, repr=False)

    def compile_named_binds(
        self, sql: str, binds: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        args: list[Any] = []

        def _repl(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in binds:
                raise ExecuteError(f"missing bind ${key} for retriever query")
            args.append(binds[key])
            return self.placeholder(len(args))

        return _NAMED_BIND.sub(_repl, sql), args

    def default_queries(self) -> dict[str, str]:
        if self._queries is None:
            raw = yaml.safe_load(self.queries_path.read_text(encoding="utf-8")) or {}
            self._queries = {str(k): str(v).strip() for k, v in raw.items() if v}
        return self._queries

    def select_entity_columns(
        self, binding: "EntityTableBinding", *, alias: str = "e"
    ) -> str:
        raise NotImplementedError

    def order_by_sql(
        self,
        binding: "EntityTableBinding",
        columns: list[Any],
        *,
        alias: str = "e",
    ) -> str:
        raise NotImplementedError

    def render_entity_by_ids(
        self,
        renderer: Any,
        binding: "EntityTableBinding",
        ids: list[str],
    ) -> tuple[str, list[Any]]:
        raise NotImplementedError


def _common_select_tail(
    binding: "EntityTableBinding", *, alias: str, attrs_sql: str | None
) -> list[str]:
    """Shared SELECT pieces after entity_id / attrs."""
    parts: list[str] = [f"{alias}.{binding.id_column} AS entity_id"]
    if attrs_sql is not None:
        parts.append(attrs_sql)
    if binding.created_at_column:
        parts.append(f"{alias}.{binding.created_at_column} AS created_at")
    if binding.popular_rank_column:
        parts.append(f"{alias}.{binding.popular_rank_column} AS popular_rank")
    return parts


def _iter_order_columns(columns: list[Any]):
    for c in columns:
        cname = c.name if hasattr(c, "name") else c["name"]
        asc = c.ascending if hasattr(c, "ascending") else c.get("ascending", True)
        nulls_first = (
            c.nulls_first if hasattr(c, "nulls_first") else c.get("nulls_first", False)
        )
        yield str(cname), bool(asc), bool(nulls_first)


def register_sql_dialect(dialect: SqlDialect, *aliases: str) -> SqlDialect:
    """Register ``dialect`` under ``dialect.name`` and optional aliases."""
    names = {str(dialect.name).lower(), *(a.lower() for a in aliases)}
    extra = getattr(dialect, "aliases", ()) or ()
    names.update(str(a).lower() for a in extra)
    for key in names:
        _REGISTRY[key] = dialect
    return dialect


def normalize_backend_name(backend: str) -> str:
    """Lowercase / strip a backend key. Does not resolve aliases or invent a default."""
    return (backend or "").strip().lower()


def canonical_backend_name(backend: str) -> str:
    """Resolve pack-owned aliases to the dialect's canonical ``name``.

    Alias maps live on installed packs (``register_sql_dialect`` / entry points),
    not in core. Unknown keys are returned lowercased so callers can still look
    up backend entry points by the raw name.
    """
    key = normalize_backend_name(backend)
    if not key:
        raise ExecuteError("backend name is required")
    if key in ("mock", "memory", "inmemory"):
        return "mock"
    _load_dialect_entry_points()
    if key in _REGISTRY:
        return str(_REGISTRY[key].name).lower()
    return key


def get_sql_dialect(backend: str) -> SqlDialect:
    """Resolve the SQL dialect for ``backend`` (loads installed plugin packs)."""
    key = normalize_backend_name(backend)
    if not key:
        raise ExecuteError("backend name is required")
    if key not in _REGISTRY:
        _load_dialect_entry_points()
    if key not in _REGISTRY:
        # Alias may already be loaded under another key; try canonical.
        try:
            canon = canonical_backend_name(key)
        except ExecuteError:
            canon = key
        if canon in _REGISTRY:
            return _REGISTRY[canon]
        raise ExecuteError(
            f"no SQL dialect registered for backend={backend!r}. "
            f"Install a pack that registers recql.dialects for this backend."
        )
    return _REGISTRY[key]


def _load_dialect_entry_points() -> None:
    """Load ``recql.dialects`` entry points so plugin packs register dialects."""
    from importlib.metadata import entry_points

    eps = entry_points()
    selected = (
        eps.select(group="recql.dialects")
        if hasattr(eps, "select")
        else eps.get("recql.dialects", [])
    )
    for ep in selected:
        try:
            loaded = ep.load()
        except Exception:
            continue
        # Prefer an explicit register() callable; importing the module is enough
        # when the dialect self-registers at import time.
        if callable(loaded):
            try:
                loaded()
            except TypeError:
                pass


def compile_named_binds(
    sql: str, binds: dict[str, Any], *, backend: str
) -> tuple[str, list[Any]]:
    """Dispatch named-bind compilation to the plugin dialect for ``backend``."""
    return get_sql_dialect(backend).compile_named_binds(sql, binds)


@lru_cache(maxsize=8)
def load_default_queries(backend: str) -> dict[str, str]:
    """Load bundled default retriever SQL from the plugin dialect.

    Non-SQL packs (MongoDB, FAISS, …) have no dialect — returns ``{}``.
    """
    try:
        return dict(get_sql_dialect(backend).default_queries())
    except ExecuteError:
        return {}
