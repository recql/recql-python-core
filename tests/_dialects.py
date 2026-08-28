"""Helpers for core tests that need an installed SQL dialect pack."""

from __future__ import annotations

import pytest

from recql.plugins.dialect import get_sql_dialect


def installed_dialect_names() -> list[str]:
    from importlib.metadata import entry_points

    eps = entry_points()
    selected = (
        eps.select(group="recql.dialects")
        if hasattr(eps, "select")
        else eps.get("recql.dialects", [])
    )
    names: list[str] = []
    for ep in selected:
        try:
            ep.load()
            d = get_sql_dialect(ep.name)
            names.append(str(d.name).lower())
        except Exception:
            continue
    return sorted(set(names))


def require_any_dialect() -> str:
    """Return one installed dialect canonical name, or skip the test."""
    names = installed_dialect_names()
    if not names:
        pytest.skip("no recql.dialects packs installed")
    return names[0]
