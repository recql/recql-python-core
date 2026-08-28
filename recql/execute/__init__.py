"""Async RankQueryConfig-driven pipeline executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recql.bind import BoundRankQuery
from recql.execute.merge import Candidate, RetrieveBag, union_dedupe
from recql.openapi_ir import rank_query_config_to_dict
from recql.plugins.base import PluginRegistry, RetrieveRequest
from recql.pyutils import gather_with_cancel


@dataclass
class ResultPage:
    items: list[Candidate]
    limit: int | None = None
    offset: int | None = None
    diagnostics: list[str] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [c.id for c in self.items]


async def execute(
    bound: BoundRankQuery,
    registry: PluginRegistry,
    *,
    pagination_key: str | None = None,
    timeout_s: float | None = None,
) -> ResultPage:
    """Execute bound RankQueryConfig: retrieve → merge → filter → score → reorder → limit."""
    import asyncio
    import time

    t0 = time.perf_counter()
    if timeout_s is not None and timeout_s > 0:
        try:
            page = await asyncio.wait_for(
                _execute_inner(bound, registry, pagination_key=pagination_key),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as e:
            from recql.errors import ExecuteError

            raise ExecuteError(f"query timeout after {timeout_s}s") from e
    else:
        page = await _execute_inner(bound, registry, pagination_key=pagination_key)
    page.diagnostics.append(f"elapsed_ms={int((time.perf_counter() - t0) * 1000)}")
    return page


async def _execute_inner(
    bound: BoundRankQuery,
    registry: PluginRegistry,
    *,
    pagination_key: str | None = None,
) -> ResultPage:
    """Inner pipeline without timeout wrapper."""
    cfg = bound.config
    cfg_dict = rank_query_config_to_dict(cfg)
    entity = cfg_dict.get("from") or "item"
    params = bound.params
    diagnostics = list(bound.diagnostics)

    steps = list(cfg.retrieve or [])
    # E-1: parallel retrieve
    coros = []
    for step in steps:
        rtype = getattr(step, "type", None)
        retriever = registry.get_retriever(str(rtype))
        where = getattr(step, "where", None)
        if where is not None and not retriever.supports_prefilter(where):
            from recql.errors import ExecuteError

            raise ExecuteError(
                f"prefilter where= cannot be enforced by {rtype} backend; fail closed"
            )
        req = RetrieveRequest(
            step=step,
            params=params,
            entity_type=str(entity),
            prefilter=where,
            bindings=getattr(registry, "_recql_bindings", None),
            catalog=bound.catalog,
        )
        coros.append(retriever.retrieve(req))

    bags: list[RetrieveBag]
    if coros:
        bags = await gather_with_cancel(*coros)
    else:
        bags = []

    # Ensure bag names follow step order / declared names
    named_bags: list[RetrieveBag] = []
    for step, bag in zip(steps, bags, strict=True):
        name = getattr(step, "name", None) or bag.name
        named_bags.append(RetrieveBag(name=str(name), candidates=bag.candidates))

    diagnostics.append(f"retrieve_bags={len(named_bags)}")
    merged = union_dedupe(named_bags)
    diagnostics.append(f"merged={len(merged)}")

    # Postfilter (global filter) — E-3/E-4: never conflate with prefilter
    filters = cfg.filter or []
    for fstep in filters:
        ftype = getattr(fstep, "type", "expression")
        plugin = registry.filters.get(str(ftype))
        if plugin is None:
            diagnostics.append(f"warning: no filter plugin for {ftype}")
            continue
        if ftype == "truncate":
            max_len = getattr(fstep, "max_length", 500)
            merged = merged[: int(max_len)]
        else:
            merged = await plugin.apply(fstep, merged, {"params": params})

    # Pagination exclusion
    if pagination_key and registry.kv is not None:
        seen = await registry.kv.load_seen(pagination_key)
        merged = [c for c in merged if c.id not in seen]
        diagnostics.append(f"pagination_excluded={len(seen)}")

    # Score
    score = cfg.score
    if score is not None:
        stype = getattr(score, "type", "score_ensemble")
        scorer = registry.scorers.get(str(stype))
        if scorer is not None:
            scores = await scorer.score_many(score, merged, {"params": params})
            alias = getattr(score, "output_alias", None) or "score"
            for c, s in zip(merged, scores, strict=True):
                c.attributes[alias] = s
            preserve = bool(getattr(score, "preserve_order", False))
            if not preserve:
                merged = sorted(
                    merged,
                    key=lambda c: float(c.attributes.get(alias, c.retrieval_score or 0.0)),
                    reverse=True,
                )

    # Reorder — language builtins first; registry only for custom / overrides
    from recql.reorder import builtin_reorderers

    builtins = builtin_reorderers()
    for rstep in cfg.reorder or []:
        rtype = getattr(rstep, "type", None)
        if rtype == "column_sort":
            cols = getattr(rstep, "columns", []) or []
            def sort_key(c: Candidate) -> tuple:
                keys = []
                for col in cols:
                    name = col.name if hasattr(col, "name") else col["name"]
                    asc = col.ascending if hasattr(col, "ascending") else col.get("ascending", True)
                    val = c.attributes.get(name, c.retrieval_score)
                    keys.append(val if asc else _negate(val))
                return tuple(keys)

            merged = sorted(merged, key=sort_key)
            continue
        key = str(rtype)
        reorderer = registry.reorderers.get(key) or builtins.get(key)
        if reorderer is None:
            diagnostics.append(f"warning: no reorderer for {rtype}")
            continue
        merged = await reorderer.apply(rstep, merged, {"params": params, "registry": registry})

    limit = cfg.limit
    offset = cfg.offset or 0
    page = merged[int(offset) :]
    if limit is not None:
        page = page[: int(limit)]

    if pagination_key and registry.kv is not None:
        ttl = 3600
        if bound.catalog and bound.catalog.deployment:
            pag = bound.catalog.deployment.get("pagination") or {}
            ttl = int(pag.get("page_expiration_in_seconds") or ttl)
        await registry.kv.remember(pagination_key, [c.id for c in page], ttl)

    return ResultPage(items=page, limit=limit, offset=offset, diagnostics=diagnostics)


def _negate(val: Any) -> Any:
    if val is None:
        return val
    if isinstance(val, (int, float)):
        return -val
    return val
