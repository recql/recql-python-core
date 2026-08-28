"""Prefilter shape classification (language-level, backend-agnostic).

Each SQL/library plugin pack declares its own capability matrix and calls
``supports_prefilter`` / ``assert_pushdown_or_raise`` with that matrix.
Unsupported ``where=`` shapes fail closed — never silently ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from recql.language import ast as A


class PrefilterShape(str, Enum):
    NONE = "none"
    EQUALITY = "equality"
    RANGE = "range"
    IN_LIST = "in"
    AND_OR = "and_or"
    LIKE = "like"
    IS_NULL = "is_null"
    FUNCTION = "function"
    ARBITRARY = "arbitrary"


@dataclass(frozen=True)
class PushdownCapability:
    retriever: str
    shapes: frozenset[PrefilterShape]
    notes: str = ""

    def allows(self, shape: PrefilterShape) -> bool:
        if PrefilterShape.ARBITRARY in self.shapes:
            return True
        if shape == PrefilterShape.NONE:
            return True
        return shape in self.shapes


_EQ = re.compile(r"(?i)\b[\w.]+\s*(=|!=|<>)\s*")
_RANGE = re.compile(r"(?i)\b(BETWEEN|>|>=|<|<=)\b")
_IN = re.compile(r"(?i)\bIN\s*[\(\[]")
_LIKE = re.compile(r"(?i)\b(LIKE|ILIKE)\b")
_IS_NULL = re.compile(r"(?i)\bIS\s+(NOT\s+)?NULL\b")
_FUNC = re.compile(r"(?i)\b[a-z_][\w]*\s*\(")
_BOOL = re.compile(r"(?i)\b(AND|OR)\b")


def classify_prefilter(expr: A.Expr | str | None) -> PrefilterShape:
    if expr is None:
        return PrefilterShape.NONE
    if isinstance(expr, A.Expr):
        text = _ast_probe(expr)
    else:
        text = str(expr)
    if not text.strip():
        return PrefilterShape.NONE
    if _LIKE.search(text):
        return PrefilterShape.LIKE
    if _IS_NULL.search(text):
        return PrefilterShape.IS_NULL
    if _IN.search(text):
        return PrefilterShape.IN_LIST
    if _RANGE.search(text):
        return PrefilterShape.RANGE
    if _BOOL.search(text):
        return PrefilterShape.AND_OR
    if _EQ.search(text):
        return PrefilterShape.EQUALITY
    if _FUNC.search(text):
        return PrefilterShape.FUNCTION
    return PrefilterShape.FUNCTION


def _ast_probe(expr: A.Expr) -> str:
    from recql.lower import expr_to_string

    try:
        return expr_to_string(expr)
    except Exception:
        return type(expr).__name__


def supports_prefilter(
    matrix: dict[str, PushdownCapability],
    retriever_type: str,
    expr: A.Expr | str | None,
) -> bool:
    cap = matrix.get(retriever_type)
    if cap is None:
        return expr is None
    return cap.allows(classify_prefilter(expr))


def assert_pushdown_or_raise(
    matrix: dict[str, PushdownCapability],
    retriever_type: str,
    expr: A.Expr | str | None,
) -> None:
    from recql.errors import ExecuteError

    if supports_prefilter(matrix, retriever_type, expr):
        return
    shape = classify_prefilter(expr)
    raise ExecuteError(
        f"prefilter where= shape={shape.value} cannot be enforced by "
        f"{retriever_type} backend; fail closed"
    )
