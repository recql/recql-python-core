"""Language built-in reorder stages (executor-owned, backend-agnostic).

``diversity`` / ``exploration`` / ``boosted`` are RecQL builtins — the executor
resolves them directly. Plugins do not need to register them. Exploration /
boosted still call registry *retrievers* when a secondary bag is needed.
``PluginRegistry.reorderers`` is only for custom / override types.
"""

from __future__ import annotations

from typing import Any

from recql.execute.merge import Candidate
from recql.plugins.base import PluginRegistry, Reorderer, RetrieveRequest


def builtin_reorderers() -> dict[str, Reorderer]:
    """Canonical built-in reorderers keyed by OpenAPI ``type``."""
    return {
        "diversity": DiversityReorderer(),
        "exploration": ExplorationReorderer(),
        "boosted": BoostedReorderer(),
    }


def _resolve_strength(step: Any, ctx: dict[str, Any], default: float = 0.5) -> float:
    raw = getattr(step, "strength", default)
    if raw is None:
        return default
    if isinstance(raw, str) and raw.startswith("$"):
        params = ctx.get("params") or {}
        body = raw[1:]
        for prefix in ("parameter.", "param.", "params."):
            if body.lower().startswith(prefix):
                body = body[len(prefix) :]
                break
        val = params.get(body, params.get(body.split(".")[-1], default))
        return float(val)
    return float(raw)


class DiversityReorderer(Reorderer):
    """Greedy MMR-style reorder on attribute Jaccard / score blend."""

    async def apply(
        self, step: Any, rows: list[Candidate], ctx: dict[str, Any]
    ) -> list[Candidate]:
        if not rows:
            return rows
        strength = _resolve_strength(step, ctx)
        cap = int(getattr(step, "max_diversity_candidates", 1000) or 1000)
        head = rows[:cap]
        tail = rows[cap:]
        selected: list[Candidate] = []
        remaining = list(head)
        while remaining:
            best_i = 0
            best_score = float("-inf")
            for i, cand in enumerate(remaining):
                rel = float(cand.retrieval_score or 0.0)
                if selected:
                    sim = max(_jaccard(cand, s) for s in selected)
                else:
                    sim = 0.0
                mmr = (1.0 - strength) * rel - strength * sim
                if mmr > best_score:
                    best_score = mmr
                    best_i = i
            selected.append(remaining.pop(best_i))
        return selected + tail


class ExplorationReorderer(Reorderer):
    async def apply(
        self, step: Any, rows: list[Candidate], ctx: dict[str, Any]
    ) -> list[Candidate]:
        strength = _resolve_strength(step, ctx)
        registry: PluginRegistry | None = ctx.get("registry")
        explore: list[Candidate] = []
        if registry is not None:
            ret = getattr(step, "retriever", None)
            if ret is not None:
                rtype = getattr(ret, "type", None)
                retriever = registry.retrievers.get(str(rtype))
                if retriever is not None:
                    bag = await retriever.retrieve(
                        RetrieveRequest(
                            step=ret,
                            params=ctx.get("params") or {},
                            bindings=getattr(registry, "_recql_bindings", None),
                            catalog=getattr(registry, "_recql_catalog", None),
                        )
                    )
                    explore = bag.candidates
        return _interleave(rows, explore, strength)


class BoostedReorderer(Reorderer):
    async def apply(
        self, step: Any, rows: list[Candidate], ctx: dict[str, Any]
    ) -> list[Candidate]:
        return await ExplorationReorderer().apply(step, rows, ctx)


def _jaccard(a: Candidate, b: Candidate) -> float:
    from recql.pyutils.jaccard import jaccard

    sa = set(str(v) for v in a.attributes.values()) | {a.id}
    sb = set(str(v) for v in b.attributes.values()) | {b.id}
    return jaccard(sa, sb)


def _interleave(
    primary: list[Candidate], secondary: list[Candidate], strength: float
) -> list[Candidate]:
    if not secondary or strength <= 0:
        return primary
    out: list[Candidate] = []
    seen: set[str] = set()
    pi = si = 0
    while pi < len(primary) or si < len(secondary):
        take_sec = (si / max(si + pi, 1)) < strength and si < len(secondary)
        if take_sec:
            c = secondary[si]
            si += 1
        elif pi < len(primary):
            c = primary[pi]
            pi += 1
        else:
            c = secondary[si]
            si += 1
        if c.id in seen:
            continue
        seen.add(c.id)
        out.append(c)
    return out


__all__ = [
    "BoostedReorderer",
    "DiversityReorderer",
    "ExplorationReorderer",
    "builtin_reorderers",
]
