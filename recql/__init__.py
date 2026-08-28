"""RecQL — Recommender Query Language (SIDE Engine Query Language)."""

from __future__ import annotations

from recql.harness import recql, recql_sync, recql_to_rank_query_config

__all__ = [
    "recql",
    "recql_sync",
    "recql_to_rank_query_config",
]

__version__ = "0.1.0"
