"""Candidate rows and §2.2 union/dedupe merge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    id: str
    retrieval_score: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    # stashed per-bag: retrieval.<name> / retrieval.<name>_rank
    retrieval: dict[str, float | int | None] = field(default_factory=dict)
    bags: list[str] = field(default_factory=list)

    def get_score(self, name: str, default: float | None = None) -> float | None:
        key = name
        if key in self.retrieval:
            val = self.retrieval[key]
            return float(val) if val is not None else default
        return default

    def get_rank(self, name: str, default: int | None = None) -> int | None:
        key = f"{name}_rank"
        if key in self.retrieval:
            val = self.retrieval[key]
            return int(val) if val is not None else default
        return default


@dataclass
class RetrieveBag:
    """One retriever's result bag (ordered)."""

    name: str
    candidates: list[Candidate]


def union_dedupe(bags: list[RetrieveBag]) -> list[Candidate]:
    """§2.2 merge: query-order stable, first non-null default retrieval_score.

    - Union by id; provenance records every bag that produced it.
    - Always stash retrieval.<name> / retrieval.<name>_rank when bag returned id.
    - Default retrieval_score: walk bags in query order; first non-null.
    - Emission order: first-seen across bags (bag 0..n, then rank within bag).
    """
    by_id: dict[str, Candidate] = {}
    order: list[str] = []

    for bag in bags:
        for rank_1based, cand in enumerate(bag.candidates, start=1):
            eid = cand.id
            score = cand.retrieval_score
            if eid not in by_id:
                merged = Candidate(
                    id=eid,
                    retrieval_score=None,
                    attributes=dict(cand.attributes),
                    retrieval={},
                    bags=[],
                )
                by_id[eid] = merged
                order.append(eid)
            else:
                merged = by_id[eid]
                # merge attributes (first wins for keys)
                for k, v in cand.attributes.items():
                    merged.attributes.setdefault(k, v)

            merged.bags.append(bag.name)
            merged.retrieval[bag.name] = score
            merged.retrieval[f"{bag.name}_rank"] = rank_1based

    # Default retrieval_score: query-order first non-null
    bag_names = [b.name for b in bags]
    result: list[Candidate] = []
    for eid in order:
        merged = by_id[eid]
        default: float | None = None
        for bname in bag_names:
            if bname in merged.retrieval and merged.retrieval[bname] is not None:
                default = float(merged.retrieval[bname])  # type: ignore[arg-type]
                break
        merged.retrieval_score = default
        result.append(merged)
    return result
