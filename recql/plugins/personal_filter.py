"""Config-driven ``personal_filter`` helpers for ``prebuilt(...)`` steps."""

from __future__ import annotations

from typing import Any

from recql.catalog.bindings import DataBindings, PersonalFilterBinding
from recql.errors import ExecuteError
from recql.plugins.dialect import canonical_backend_name, get_sql_dialect


def resolve_param(val: Any, params: dict[str, Any]) -> Any:
    if isinstance(val, str) and val.startswith("$"):
        body = val[1:]
        for prefix in ("parameter.", "param.", "params."):
            if body.lower().startswith(prefix):
                body = body[len(prefix) :]
                break
        if body in params:
            return params[body]
        short = body.split(".")[-1]
        if short in params:
            return params[short]
    return val


def lookup_personal_filter(
    bindings: DataBindings, filter_ref: Any
) -> PersonalFilterBinding | None:
    if filter_ref is None:
        return None
    return bindings.personal_filter(str(filter_ref))


def render_personal_filter_ids(
    bindings: DataBindings,
    filt: PersonalFilterBinding,
    *,
    entity_id: str,
    by_user: bool,
) -> tuple[str, list[Any]]:
    """SQL that returns banned ids for a user- or item-keyed personal filter."""
    if by_user:
        select_column = filt.item_id_column or "item_id"
        where_column = filt.user_id_column or "user_id"
    else:
        select_column = filt.user_id_column or "user_id"
        where_column = filt.item_id_column or "item_id"
    structural = {
        "filter_from": filt.from_sql,
        "select_column": select_column,
        "where_column": where_column,
        # Backward-compatible aliases for interaction_items_for_user
        "interaction_from": filt.from_sql,
        "item_id_column": select_column if by_user else (filt.item_id_column or "item_id"),
        "user_id_column": where_column if by_user else (filt.user_id_column or "user_id"),
    }
    binds = {"entity_id": str(entity_id)}
    backend = bindings.backend
    if not (backend or "").strip():
        raise ExecuteError("DataBindings.backend is required (no default backend)")
    backend = canonical_backend_name(backend)
    dialect = get_sql_dialect(backend)
    defaults = dialect.default_queries()

    template = (
        filt.queries.get("personal_filter_ids")
        or bindings.query_templates.get("personal_filter_ids")
        or defaults.get("personal_filter_ids")
    )
    if template is None and by_user:
        template = (
            filt.queries.get("interaction_items_for_user")
            or bindings.query_templates.get("interaction_items_for_user")
            or defaults.get("interaction_items_for_user")
        )
    if not template:
        raise ExecuteError(
            f"no personal_filter query template for filter={filt.name!r}"
        )
    try:
        sql = template.format(**structural)
    except KeyError as exc:
        raise ExecuteError(
            f"personal_filter template missing structural key: {exc}"
        ) from exc
    return dialect.compile_named_binds(sql, binds)


def ban_ids_from_rows(rows: list[dict[str, Any]]) -> set[str]:
    ban: set[str] = set()
    for r in rows:
        for key in ("ban_id", "item_id", "user_id", "entity_id"):
            if key in r and r[key] is not None:
                ban.add(str(r[key]))
                break
        else:
            if r:
                ban.add(str(next(iter(r.values()))))
    return ban
