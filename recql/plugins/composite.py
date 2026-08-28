"""Composite and routed plugin registry for multi-backend RecQL engines."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from recql.errors import ExecuteError
from recql.execute.merge import Candidate, RetrieveBag
from recql.language import ast as A
from recql.plugins.base import (
    FilterPlugin,
    KvStore,
    PluginRegistry,
    Reorderer,
    RetrieveRequest,
    Retriever,
    Scorer,
)
from recql.plugins.dialect import canonical_backend_name, normalize_backend_name


class RoutedRetriever(Retriever):
    """Base class for retrievers that route to a target backend registry."""

    def __init__(
        self,
        registries: dict[str, PluginRegistry],
        retriever_type: str,
        default_backend: str | None = None,
    ) -> None:
        self.registries = registries
        self.retriever_type = retriever_type
        self.default_backend = default_backend

    def _resolve_backend_name(self, target: str | None) -> str | None:
        if not target:
            return self.default_backend
        if target in self.registries:
            return target
        key = normalize_backend_name(target)
        if key in self.registries:
            return key
        try:
            canon = canonical_backend_name(target)
            if canon in self.registries:
                return canon
        except Exception:
            pass
        return self.default_backend

    def resolve_registry(self, target: str | None) -> PluginRegistry:
        bname = self._resolve_backend_name(target)
        if bname and bname in self.registries:
            return self.registries[bname]
        if self.default_backend and self.default_backend in self.registries:
            return self.registries[self.default_backend]
        if self.registries:
            return next(iter(self.registries.values()))
        raise ExecuteError(f"No plugin registries available for routing (requested {target!r})")

    def target_for_request(self, req: RetrieveRequest) -> str | None:
        step = req.step
        # 1. Step-level explicit backend / engine_ref override
        target = getattr(step, "backend", None) or getattr(step, "engine_ref", None)
        if target:
            return str(target)

        cat = req.catalog
        bindings = req.bindings

        # 2. Embedding-based target (for similarity)
        emb_ref = getattr(step, "embedding_ref", None)
        if emb_ref:
            if cat is not None:
                b = cat.backend_for_embedding(str(emb_ref))
                if b:
                    return b
            if bindings is not None:
                b = bindings.backend_for_embedding(str(emb_ref), req.entity_type)
                if b:
                    return b

        # 3. Entity-based target
        if cat is not None:
            b = cat.backend_for_entity(req.entity_type)
            if b:
                return b

        return self.default_backend

    def supports_prefilter(self, expr: A.Expr | str | None) -> bool:
        if expr is None:
            return True
        # Check if at least default or any registry supports it
        for reg in self.registries.values():
            if self.retriever_type in reg.retrievers:
                if not reg.retrievers[self.retriever_type].supports_prefilter(expr):
                    return False
        return True

    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        target = self.target_for_request(req)
        reg = self.resolve_registry(target)
        retriever = reg.get_retriever(self.retriever_type)
        return await retriever.retrieve(req)


class RoutedSimilarityRetriever(RoutedRetriever):
    def __init__(
        self,
        registries: dict[str, PluginRegistry],
        default_backend: str | None = None,
    ) -> None:
        super().__init__(registries, "similarity", default_backend)

    def target_for_request(self, req: RetrieveRequest) -> str | None:
        step = req.step
        target = getattr(step, "backend", None) or getattr(step, "engine_ref", None)
        if target:
            return str(target)
        emb_ref = getattr(step, "embedding_ref", None)
        if emb_ref:
            if req.catalog is not None:
                b = req.catalog.backend_for_embedding(str(emb_ref))
                if b:
                    return b
            if req.bindings is not None:
                b = req.bindings.backend_for_embedding(str(emb_ref), req.entity_type)
                if b:
                    return b
        return self.default_backend


class RoutedTextSearchRetriever(RoutedRetriever):
    def __init__(
        self,
        registries: dict[str, PluginRegistry],
        default_backend: str | None = None,
    ) -> None:
        super().__init__(registries, "text_search", default_backend)

    def target_for_request(self, req: RetrieveRequest) -> str | None:
        step = req.step
        target = getattr(step, "backend", None) or getattr(step, "engine_ref", None)
        if target:
            return str(target)

        mode = getattr(step, "mode", None)
        mode_str = str(getattr(mode, "type", mode) or "").lower()

        # If vector mode, route via the text embedding
        if "vector" in mode_str:
            text_emb = getattr(mode, "text_embedding_ref", None) or "content_embedding"
            if req.catalog is not None:
                b = req.catalog.backend_for_embedding(str(text_emb))
                if b:
                    return b
                b = req.catalog.backend_for_lexical()
                if b:
                    return b
            if req.bindings is not None:
                b = req.bindings.backend_for_embedding(str(text_emb), req.entity_type)
                if b:
                    return b
        else:
            # Lexical / BM25 / Text search mode
            if req.catalog is not None:
                b = req.catalog.backend_for_lexical()
                if b:
                    return b

        return self.default_backend


class RoutedScorer(Scorer):
    """Routes score_many calls to the appropriate backend for a model or feature store."""

    def __init__(
        self,
        registries: dict[str, PluginRegistry],
        default_backend: str | None = None,
    ) -> None:
        self.registries = registries
        self.default_backend = default_backend

    def resolve_registry(self, target: str | None) -> PluginRegistry:
        if target and target in self.registries:
            return self.registries[target]
        if self.default_backend and self.default_backend in self.registries:
            return self.registries[self.default_backend]
        return next(iter(self.registries.values()))

    async def score_many(
        self, plan: Any, candidates: list[Candidate], ctx: dict[str, Any]
    ) -> list[float]:
        model_name = (
            getattr(plan, "value_model", None)
            or getattr(plan, "model", None)
            or getattr(plan, "name", None)
        )
        target = None
        cat = ctx.get("catalog")
        if cat and model_name:
            target = cat.backend_for_model(str(model_name))
        reg = self.resolve_registry(target)
        scorer_type = getattr(plan, "type", "score_ensemble") or "score_ensemble"
        if scorer_type in reg.scorers:
            return await reg.scorers[scorer_type].score_many(plan, candidates, ctx)
        if "score_ensemble" in reg.scorers:
            return await reg.scorers["score_ensemble"].score_many(plan, candidates, ctx)
        if "score" in reg.scorers:
            return await reg.scorers["score"].score_many(plan, candidates, ctx)
        if "model" in reg.scorers:
            return await reg.scorers["model"].score_many(plan, candidates, ctx)
        return [float(c.retrieval_score or 0.0) for c in candidates]


class RoutedFilterPlugin(FilterPlugin):
    """Routes prebuilt filter plugins to the backend owning the filter relation."""

    def __init__(
        self,
        registries: dict[str, PluginRegistry],
        filter_name: str | None = None,
        default_backend: str | None = None,
    ) -> None:
        self.registries = registries
        self.filter_name = filter_name
        self.default_backend = default_backend

    async def apply(
        self, step: Any, rows: list[Candidate], ctx: dict[str, Any]
    ) -> list[Candidate]:
        fname = self.filter_name or getattr(step, "filter_ref", None) or getattr(step, "name", None)
        fname_str = str(fname) if fname else "exclude_seen"
        cat = ctx.get("catalog")
        target = None
        if cat and hasattr(cat, "backend_for_filter"):
            target = cat.backend_for_filter(fname_str)
        bname = target or self.default_backend
        reg = self.registries.get(bname) if bname else next(iter(self.registries.values()))
        if reg:
            if "prebuilt" in reg.filters:
                return await reg.filters["prebuilt"].apply(step, rows, ctx)
            if fname_str in reg.filters:
                return await reg.filters[fname_str].apply(step, rows, ctx)
        return rows


class CompositePluginRegistry(PluginRegistry):
    """PluginRegistry composed of multiple backend PluginRegistries."""

    def __init__(
        self,
        registries: dict[str, PluginRegistry],
        default_backend: str | None = None,
        kv: KvStore | None = None,
    ) -> None:
        super().__init__()
        self.registries = registries
        self.default_backend = default_backend or (
            next(iter(registries.keys())) if registries else None
        )

        # Primary KV store
        if kv is not None:
            self.kv = kv
        elif self.default_backend and self.default_backend in self.registries:
            self.kv = self.registries[self.default_backend].kv
        else:
            for r in self.registries.values():
                if r.kv is not None:
                    self.kv = r.kv
                    break

        # Wire routed retrievers
        self.retrievers["similarity"] = RoutedSimilarityRetriever(
            self.registries, default_backend=self.default_backend
        )
        self.retrievers["text_search"] = RoutedTextSearchRetriever(
            self.registries, default_backend=self.default_backend
        )
        self.retrievers["column_order"] = RoutedRetriever(
            self.registries, "column_order", default_backend=self.default_backend
        )
        self.retrievers["filter"] = RoutedRetriever(
            self.registries, "filter", default_backend=self.default_backend
        )
        self.retrievers["candidate_ids"] = RoutedRetriever(
            self.registries, "candidate_ids", default_backend=self.default_backend
        )
        self.retrievers["candidate_attributes"] = RoutedRetriever(
            self.registries, "candidate_attributes", default_backend=self.default_backend
        )

        # Merge any custom retrievers from underlying registries
        for reg in self.registries.values():
            for rk, rv in reg.retrievers.items():
                if rk not in self.retrievers:
                    self.retrievers[rk] = RoutedRetriever(
                        self.registries, rk, default_backend=self.default_backend
                    )

        # Wire routed scorer
        self.scorers["score"] = RoutedScorer(
            self.registries, default_backend=self.default_backend
        )
        self.scorers["score_ensemble"] = self.scorers["score"]
        self.scorers["model"] = self.scorers["score"]

        # Merge reorderers (in-memory algorithms)
        for reg in self.registries.values():
            for k, reorderer in reg.reorderers.items():
                if k not in self.reorderers:
                    self.reorderers[k] = reorderer

        # Wire routed personal/prebuilt filters
        self.filters["prebuilt"] = RoutedFilterPlugin(
            self.registries, default_backend=self.default_backend
        )
        all_filters: set[str] = set()
        for reg in self.registries.values():
            all_filters.update(reg.filters.keys())
        for fname in all_filters:
            self.filters[fname] = RoutedFilterPlugin(
                self.registries, filter_name=fname, default_backend=self.default_backend
            )
