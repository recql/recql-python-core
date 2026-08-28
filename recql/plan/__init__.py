"""Bound config dump helpers."""

from __future__ import annotations

from typing import Any

from recql.bind import BoundRankQuery
from recql.openapi_ir import rank_query_config_to_dict


def dump_bound(bound: BoundRankQuery) -> dict[str, Any]:
    return {
        "config": rank_query_config_to_dict(bound.config),
        "diagnostics": bound.diagnostics,
        "params": bound.params,
    }
