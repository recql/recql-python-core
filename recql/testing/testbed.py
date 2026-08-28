"""Backend testbed contract for the shared conformance suite.

Each database pack provides a ``recql_testbed`` pytest fixture that yields a
``RecqlTestbed``. Core owns the assertions; packs only open a DB, seed data,
and expose a registry + catalog.

Capability flags (see ``recql.testing.features``) gate tests so ANN-only packs
like FAISS skip lexical / scoring / SQL-dialect checks.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recql.catalog import EngineCatalog
from recql.plugins.base import PluginRegistry
from recql.testing.features import SQL_BACKEND_FEATURES

_FIXTURES = Path(__file__).resolve().parent / "fixtures"


@dataclass
class RecqlTestbed:
    """Live backend under test — same suite runs for every conforming pack."""

    backend: str
    registry: PluginRegistry
    catalog: EngineCatalog
    """Engine catalog matching the seeded schema (column names, embeddings, filters)."""

    dims: int = 8
    popular_rank_column: str = "_derived_popular_rank"
    """Physical column used in ``column_order`` SQL (differs per seed schema)."""

    features: frozenset[str] = field(default_factory=lambda: frozenset(SQL_BACKEND_FEATURES))
    """Capability flags — see ``recql.testing.features``."""

    # Soft ANN agreement vs brute-force fake embeddings (deterministic CI).
    ann_overlap_k: int = 10
    ann_min_overlap: float = 0.5
    """Require overlap@k ≥ this fraction vs exact cosine ranking (default 50%)."""

    close: Callable[[], Awaitable[None]] | None = None
    """Optional cleanup (pool close). Called by the pack's fixture teardown."""

    async def popular_item_ids(self, limit: int) -> list[str] | None:
        """Optional ground-truth popular ranking; ``None`` skips strict id checks."""
        return None

    def ann_corpus(self) -> list[dict[str, str]]:
        """Item texts for brute-force reference ranks (override if seed differs)."""
        from recql.testing.ann import load_ann_corpus

        return load_ann_corpus()

    def query_fixture(self, name: str) -> str:
        path = _FIXTURES / "queries" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.read_text(encoding="utf-8")

    async def execute(
        self,
        query: object,
        *,
        params: dict[str, Any] | None = None,
        pagination_key: str | None = None,
    ):
        from recql.bind import bind
        from recql.execute import execute
        from recql.yaml_query import query_input_to_rank_query_config

        cfg = query_input_to_rank_query_config(query)
        return await execute(
            bind(cfg, self.catalog, params=params or {}),
            self.registry,
            pagination_key=pagination_key,
        )


def fixtures_dir() -> Path:
    return _FIXTURES
