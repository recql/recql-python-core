"""Shared SQL template retriever tests."""

from __future__ import annotations

import pytest

from recql.catalog.bindings import default_fixture_bindings
from recql.execute.merge import Candidate
from recql.plugins.base import RetrieveRequest
from recql.plugins.sql_common import (
    SqlPrebuiltFilter,
    TemplateColumnOrderRetriever,
    flatten_id_list,
)

from tests._dialects import require_any_dialect


class _FakeStep:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _RecordingExecutor:
    def __init__(self, rows):
        self.rows = rows
        self.calls: list[tuple[str, list]] = []

    async def fetch_all(self, sql, args=None):
        self.calls.append((sql, list(args or [])))
        return list(self.rows)


def test_flatten_id_list_nested_and_params():
    assert flatten_id_list(["a", ["b", "c"]], {}) == ["a", "b", "c"]
    assert flatten_id_list("$ids", {"ids": ["x", "y"]}) == ["x", "y"]


@pytest.mark.asyncio
async def test_template_column_order_renders_and_ranks():
    backend = require_any_dialect()
    ex = _RecordingExecutor(
        [
            {"entity_id": "1", "attrs": {"t": "a"}},
            {"entity_id": "2", "attrs": {"t": "b"}},
        ]
    )
    ret = TemplateColumnOrderRetriever(ex, default_backend=backend)
    step = _FakeStep(
        name="pop",
        limit=10,
        columns=[{"name": "_derived_popular_rank", "ascending": True}],
        where=None,
    )
    bag = await ret.retrieve(
        RetrieveRequest(
            step=step,
            bindings=default_fixture_bindings(backend=backend),
        )
    )
    assert bag.name == "pop"
    assert [c.id for c in bag.candidates] == ["1", "2"]
    assert bag.candidates[0].retrieval_score > bag.candidates[1].retrieval_score
    assert ex.calls and "ORDER BY" in ex.calls[0][0]


@pytest.mark.asyncio
async def test_sql_prebuilt_filter_excludes_seen():
    backend = require_any_dialect()
    ex = _RecordingExecutor([{"ban_id": "a"}, {"item_id": "b"}])
    filt = SqlPrebuiltFilter(
        ex,
        bindings=default_fixture_bindings(backend=backend),
        default_backend=backend,
    )
    rows = [
        Candidate(id="a", retrieval_score=1.0),
        Candidate(id="b", retrieval_score=0.9),
        Candidate(id="c", retrieval_score=0.8),
    ]
    step = _FakeStep(filter_ref="exclude_seen", input_user_id="u1")
    out = await filt.apply(step, rows, {"params": {}})
    assert [c.id for c in out] == ["c"]
