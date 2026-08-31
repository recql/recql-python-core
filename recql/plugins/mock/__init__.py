"""Deterministic in-memory mock plugins for unit tests."""

from __future__ import annotations

import asyncio
from typing import Any

from recql.errors import ExecuteError
from recql.execute.merge import Candidate, RetrieveBag
from recql.language import ast as A
from recql.expr import ExpressionScorer
from recql.plugins.base import (
    FilterPlugin,
    KvStore,
    PluginRegistry,
    RetrieveRequest,
    Retriever,
    Scorer,
)


class InMemoryKvStore(KvStore):
    def __init__(self) -> None:
        self._seen: dict[str, set[str]] = {}

    async def load_seen(self, key: str) -> set[str]:
        return set(self._seen.get(key, set()))

    async def remember(self, key: str, ids: list[str], ttl: int) -> None:
        bag = self._seen.setdefault(key, set())
        bag.update(ids)


class MockRetriever(Retriever):
    """Returns configured candidates; optional delay to shuffle finish order."""

    def __init__(
        self,
        bags_by_name: dict[str, list[tuple[str, float]]] | None = None,
        *,
        delay_by_name: dict[str, float] | None = None,
        supports_where: bool = True,
        vectors: dict[tuple[str, str, str], list[float]] | dict[str, list[float]] | None = None,
        interactions: dict[str, list[str]] | None = None,
    ) -> None:
        self.bags_by_name = bags_by_name or {}
        self.delay_by_name = delay_by_name or {}
        self.supports_where = supports_where
        self.vectors = vectors or {}
        self.interactions = interactions or {}
        self.last_query_vector: list[float] | None = None

    def supports_prefilter(self, expr: A.Expr | str | None) -> bool:
        if expr is None:
            return True
        return self.supports_where

    async def lookup_vector(
        self,
        embedding_ref: str,
        entity_type: str,
        entity_id: str,
        *,
        req: RetrieveRequest | None = None,
    ) -> list[float] | None:
        # Check (embedding_ref, entity_type, entity_id) then entity_id
        if (embedding_ref, entity_type, entity_id) in self.vectors:
            return self.vectors[(embedding_ref, entity_type, entity_id)]
        if entity_id in self.vectors:
            return self.vectors[entity_id]
        return None

    async def lookup_vectors(
        self,
        embedding_ref: str,
        entity_type: str,
        entity_ids: list[str],
        *,
        req: RetrieveRequest | None = None,
    ) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for eid in entity_ids:
            v = await self.lookup_vector(
                embedding_ref, entity_type, eid, req=req
            )
            if v is not None:
                out[eid] = v
        return out

    async def lookup_interactions(
        self,
        user_id: str,
        limit: int = 10,
        *,
        req: RetrieveRequest | None = None,
    ) -> list[str]:
        items = self.interactions.get(user_id, [])
        return items[:limit]

    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        step = req.step
        name = getattr(step, "name", None) or getattr(step, "type", "bag")
        self.last_query_vector = getattr(step, "query_vector", None) or (
            req.params.get("__query_vector__") if req.params else None
        )
        where = getattr(step, "where", None)
        if where is not None and not self.supports_prefilter(where):
            raise ExecuteError(
                f"prefilter where= not supported by mock retriever for {name}; fail closed"
            )
        delay = self.delay_by_name.get(str(name), 0.0)
        if delay:
            await asyncio.sleep(delay)
        raw = self.bags_by_name.get(str(name), [])
        limit = getattr(step, "limit", None) or len(raw)
        cands = [
            Candidate(id=i, retrieval_score=s, attributes={})
            for i, s in raw[: int(limit)]
        ]
        return RetrieveBag(name=str(name), candidates=cands)


class MockScorer(Scorer):
    async def score_many(
        self, plan: Any, candidates: list[Candidate], ctx: dict[str, Any]
    ) -> list[float]:
        # Simple: use retrieval_score or 0
        return [float(c.retrieval_score or 0.0) for c in candidates]


class MockExpressionFilter(FilterPlugin):
    async def apply(
        self, step: Any, rows: list[Candidate], ctx: dict[str, Any]
    ) -> list[Candidate]:
        # Phase A: expression filters that mention id= can filter; else pass
        expr = getattr(step, "expression", "") or ""
        if not expr:
            return rows
        # very small evaluator for tests: keep all
        return rows


def mock_registry(
    bags_by_name: dict[str, list[tuple[str, float]]],
    *,
    delay_by_name: dict[str, float] | None = None,
    supports_where: bool = True,
) -> PluginRegistry:
    retriever = MockRetriever(
        bags_by_name, delay_by_name=delay_by_name, supports_where=supports_where
    )
    return PluginRegistry(
        retrievers={
            "similarity": retriever,
            "text_search": retriever,
            "column_order": retriever,
            "filter": retriever,
            "candidate_ids": retriever,
            "candidate_attributes": retriever,
        },
        scorers={"score_ensemble": ExpressionScorer(), "passthrough": MockScorer()},
        # diversity / exploration / boosted / column_sort are executor builtins
        reorderers={},
        filters={
            "expression": MockExpressionFilter(),
            "prebuilt": MockExpressionFilter(),
            "truncate": MockExpressionFilter(),
        },
        kv=InMemoryKvStore(),
    )
