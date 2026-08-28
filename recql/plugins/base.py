"""Plugin ABCs and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from recql.execute.merge import Candidate, RetrieveBag
from recql.language import ast as A


@dataclass
class RetrieveRequest:
    step: Any  # OpenAPI retrieve step struct
    params: dict[str, Any] = field(default_factory=dict)
    entity_type: str = "item"
    prefilter: A.Expr | str | None = None
    bindings: Any | None = None  # DataBindings | None
    catalog: Any | None = None


class Retriever(ABC):
    @abstractmethod
    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        ...

    def supports_prefilter(self, expr: A.Expr | str | None) -> bool:
        return expr is None


class Scorer(ABC):
    @abstractmethod
    async def score_many(
        self, plan: Any, candidates: list[Candidate], ctx: dict[str, Any]
    ) -> list[float]:
        ...


class Reorderer(ABC):
    @abstractmethod
    async def apply(
        self, step: Any, rows: list[Candidate], ctx: dict[str, Any]
    ) -> list[Candidate]:
        ...


class KvStore(ABC):
    @abstractmethod
    async def load_seen(self, key: str) -> set[str]:
        ...

    @abstractmethod
    async def remember(self, key: str, ids: list[str], ttl: int) -> None:
        ...


class FilterPlugin(ABC):
    @abstractmethod
    async def apply(
        self, step: Any, rows: list[Candidate], ctx: dict[str, Any]
    ) -> list[Candidate]:
        ...


@dataclass
class PluginRegistry:
    retrievers: dict[str, Retriever] = field(default_factory=dict)
    scorers: dict[str, Scorer] = field(default_factory=dict)
    reorderers: dict[str, Reorderer] = field(default_factory=dict)
    filters: dict[str, FilterPlugin] = field(default_factory=dict)
    kv: KvStore | None = None

    def get_retriever(self, step_type: str) -> Retriever:
        if step_type not in self.retrievers:
            raise KeyError(f"no retriever plugin for type={step_type}")
        return self.retrievers[step_type]
