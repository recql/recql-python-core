"""Shared SQL-template retriever orchestration (backend-agnostic).

Driver I/O and dialect-specific SQL stay in plugins; this module owns the
common retrieve flow: bindings → render template → fetch → Candidates.
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Protocol

from recql.catalog.bindings import (
    DataBindings,
    bindings_from_catalog,
    default_fixture_bindings,
)
from recql.catalog.query_templates import QueryRenderer
from recql.errors import ExecuteError
from recql.execute.merge import Candidate, RetrieveBag
from recql.language import ast as A
from recql.plugins.base import FilterPlugin, RetrieveRequest, Retriever
from recql.plugins.personal_filter import (
    ban_ids_from_rows,
    lookup_personal_filter,
    render_personal_filter_ids,
    resolve_param,
)

FetchAll = Callable[[str, list[Any] | None], Awaitable[list[dict[str, Any]]]]
DbFetchAll = Callable[
    [Any, str, list[Any] | None], Awaitable[list[dict[str, Any]]]
]
PushdownAssert = Callable[[str, Any], None]
SupportsPrefilter = Callable[[str, A.Expr | str | None], bool]


class SqlExecutor(Protocol):
    async def fetch_all(
        self, sql: str, args: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        ...


class AsyncpgExecutor:
    """``SqlExecutor`` over an asyncpg pool."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def fetch_all(
        self, sql: str, args: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(sql, *(args or []))
            return [dict(r) for r in rows]


class BoundDbExecutor:
    """``SqlExecutor`` wrapping ``fetch_all(db, sql, args)`` helpers."""

    def __init__(self, db: Any, fetch_fn: DbFetchAll) -> None:
        self.db = db
        self._fetch_fn = fetch_fn

    async def fetch_all(
        self, sql: str, args: list[Any] | None = None
    ) -> list[dict[str, Any]]:
        return await self._fetch_fn(self.db, sql, args)


def attrs_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return dict(json.loads(value))
    return dict(value)


def bindings_for_request(
    req: RetrieveRequest, *, default_backend: str
) -> DataBindings:
    if req.bindings is not None:
        return req.bindings
    if req.catalog is not None:
        return bindings_from_catalog(req.catalog, backend=default_backend)
    return default_fixture_bindings(backend=default_backend)


def flatten_id_list(raw_ids: Any, params: dict[str, Any]) -> list[str]:
    """Normalize candidate_ids step input into a flat list of string ids."""
    if raw_ids is None:
        return []
    if isinstance(raw_ids, str):
        resolved = resolve_param(raw_ids, params)
        if isinstance(resolved, list):
            return [str(x) for x in resolved]
        return [str(resolved)]
    flat: list[str] = []
    for i in list(raw_ids):
        resolved = resolve_param(i, params)
        if isinstance(resolved, list):
            flat.extend(str(x) for x in resolved)
        elif resolved is not None:
            flat.append(str(resolved))
    return flat


def rows_to_ranked_candidates(
    rows: list[dict[str, Any]], *, score_from_rank: bool = True
) -> list[Candidate]:
    out: list[Candidate] = []
    n = len(rows)
    for i, r in enumerate(rows):
        score = float(n - i) if score_from_rank else 1.0
        if "score" in r and r["score"] is not None:
            score = float(r["score"])
        out.append(
            Candidate(
                id=str(r["entity_id"]),
                retrieval_score=score,
                attributes=attrs_dict(r.get("attrs")),
            )
        )
    return out


class TemplateColumnOrderRetriever(Retriever):
    def __init__(
        self,
        executor: SqlExecutor,
        *,
        default_backend: str,
        assert_pushdown: PushdownAssert | None = None,
        supports_prefilter_fn: SupportsPrefilter | None = None,
    ) -> None:
        self.executor = executor
        self.default_backend = default_backend
        self._assert_pushdown = assert_pushdown
        self._supports_prefilter = supports_prefilter_fn

    def supports_prefilter(self, expr: A.Expr | str | None) -> bool:
        if self._supports_prefilter is not None:
            return self._supports_prefilter("column_order", expr)
        return expr is None

    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        step = req.step
        name = getattr(step, "name", None) or "column_order"
        limit = int(getattr(step, "limit", 100) or 100)
        cols = list(getattr(step, "columns", []) or [])
        if not cols:
            raise ExecuteError("column_order requires columns")
        bindings = bindings_for_request(req, default_backend=self.default_backend)
        renderer = QueryRenderer(bindings)
        b = bindings.entity(req.entity_type)
        where = getattr(step, "where", None)
        if where and self._assert_pushdown is not None:
            self._assert_pushdown("column_order", where)
        structural = {
            **renderer.entity_structural(b),
            "order_by": renderer.dialect.order_by_sql(b, cols, alias="e"),
            "where": where or "TRUE",
        }
        tpl = "entity_column_order" if where else "entity_column_order_open"
        sql, args = renderer.render(
            tpl, structural=structural, binds={"limit": limit}, entity=b
        )
        rows = await self.executor.fetch_all(sql, args)
        return RetrieveBag(
            name=str(name),
            candidates=rows_to_ranked_candidates(rows, score_from_rank=True),
        )


class TemplateFilterRetriever(Retriever):
    def __init__(
        self,
        executor: SqlExecutor,
        *,
        default_backend: str,
        assert_pushdown: PushdownAssert | None = None,
        supports_prefilter_fn: SupportsPrefilter | None = None,
        default_where: str = "TRUE",
    ) -> None:
        self.executor = executor
        self.default_backend = default_backend
        self._assert_pushdown = assert_pushdown
        self._supports_prefilter = supports_prefilter_fn
        self.default_where = default_where

    def supports_prefilter(self, expr: A.Expr | str | None) -> bool:
        if self._supports_prefilter is not None:
            return self._supports_prefilter("filter", expr)
        return True

    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        step = req.step
        name = getattr(step, "name", None) or "filter"
        limit = int(getattr(step, "limit", 100) or 100)
        bindings = bindings_for_request(req, default_backend=self.default_backend)
        renderer = QueryRenderer(bindings)
        b = bindings.entity(req.entity_type)
        where = (
            getattr(step, "where", None)
            or getattr(step, "expression", None)
            or self.default_where
        )
        if self._assert_pushdown is not None:
            self._assert_pushdown("filter", where)
        sql, args = renderer.render(
            "entity_filter",
            structural={**renderer.entity_structural(b), "where": where},
            binds={"limit": limit},
            entity=b,
        )
        rows = await self.executor.fetch_all(sql, args)
        return RetrieveBag(
            name=str(name),
            candidates=rows_to_ranked_candidates(rows, score_from_rank=True),
        )


class TemplateCandidateIdsRetriever(Retriever):
    def __init__(
        self,
        executor: SqlExecutor,
        *,
        default_backend: str,
        supports_prefilter_fn: SupportsPrefilter | None = None,
        emit_missing_ids: bool = False,
    ) -> None:
        self.executor = executor
        self.default_backend = default_backend
        self._supports_prefilter = supports_prefilter_fn
        self.emit_missing_ids = emit_missing_ids

    def supports_prefilter(self, expr: A.Expr | str | None) -> bool:
        if self._supports_prefilter is not None:
            return self._supports_prefilter("candidate_ids", expr)
        return expr is None

    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        step = req.step
        name = getattr(step, "name", None) or "candidate_ids"
        raw = (
            getattr(step, "item_ids", None)
            or getattr(step, "ids", None)
            or getattr(step, "candidate_ids", None)
            or []
        )
        flat = flatten_id_list(raw, req.params or {})
        limit = getattr(step, "limit", None)
        if limit is not None:
            flat = flat[: int(limit)]
        if not flat:
            return RetrieveBag(name=str(name), candidates=[])
        bindings = bindings_for_request(req, default_backend=self.default_backend)
        renderer = QueryRenderer(bindings)
        b = bindings.entity(req.entity_type)
        sql, args = renderer.dialect.render_entity_by_ids(renderer, b, flat)
        rows = await self.executor.fetch_all(sql, args)
        by_id = {str(r["entity_id"]): r for r in rows}
        cands: list[Candidate] = []
        for i, eid in enumerate(flat):
            r = by_id.get(eid)
            if r is None:
                if self.emit_missing_ids:
                    cands.append(
                        Candidate(
                            id=eid,
                            retrieval_score=float(len(flat) - i),
                            attributes={},
                        )
                    )
                continue
            cands.append(
                Candidate(
                    id=eid,
                    retrieval_score=float(len(flat) - i),
                    attributes=attrs_dict(r.get("attrs")),
                )
            )
        return RetrieveBag(name=str(name), candidates=cands)


class SqlPrebuiltFilter(FilterPlugin):
    """``prebuilt(...)`` via ``data.filters`` personal_filter bindings."""

    def __init__(
        self,
        executor: SqlExecutor,
        *,
        default_backend: str,
        bindings: DataBindings | None = None,
    ) -> None:
        self.executor = executor
        self.bindings = bindings or default_fixture_bindings(backend=default_backend)

    async def apply(self, step, rows, ctx):
        ref = getattr(step, "filter_ref", "") or ""
        filt = lookup_personal_filter(self.bindings, ref)
        if filt is None:
            return rows
        params = ctx.get("params") or {}
        uid = resolve_param(getattr(step, "input_user_id", None), params)
        iid = resolve_param(getattr(step, "input_item_id", None), params)
        if uid is not None:
            entity_id, by_user = str(uid), True
        elif iid is not None:
            entity_id, by_user = str(iid), False
        else:
            return rows
        sql, args = render_personal_filter_ids(
            self.bindings, filt, entity_id=entity_id, by_user=by_user
        )
        seen_rows = await self.executor.fetch_all(sql, args)
        ban = ban_ids_from_rows(seen_rows)
        return [c for c in rows if c.id not in ban]
