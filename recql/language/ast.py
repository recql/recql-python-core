"""RecQL AST nodes (parse tree; not the executable IR)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Node:
    line: int = 0
    column: int = 0


@dataclass(slots=True)
class Name(Node):
    value: str = ""


@dataclass(slots=True)
class Param(Node):
    value: str = ""  # includes leading $


@dataclass(slots=True)
class Literal(Node):
    value: Any = None
    kind: str = ""  # string|int|float|bool|null


@dataclass(slots=True)
class ArrayLiteral(Node):
    elements: list[Expr] = field(default_factory=list)


@dataclass(slots=True)
class UnaryOp(Node):
    op: str = ""
    operand: Expr | None = None


@dataclass(slots=True)
class BinaryOp(Node):
    op: str = ""
    left: Expr | None = None
    right: Expr | None = None


@dataclass(slots=True)
class IsPredicate(Node):
    expr: Expr | None = None
    negated: bool = False
    target: str = ""  # NULL|TRUE|FALSE


@dataclass(slots=True)
class InPredicate(Node):
    expr: Expr | None = None
    negated: bool = False
    values: list[Expr] | None = None  # list form
    param: Param | None = None  # bare IN $param


@dataclass(slots=True)
class LikePredicate(Node):
    expr: Expr | None = None
    negated: bool = False
    case_insensitive: bool = False
    pattern: Expr | None = None


@dataclass(slots=True)
class BetweenPredicate(Node):
    expr: Expr | None = None
    negated: bool = False
    low: Expr | None = None
    high: Expr | None = None


@dataclass(slots=True)
class CallArg(Node):
    name: str | None = None  # named arg
    value: Expr | None = None


@dataclass(slots=True)
class FuncCall(Node):
    name: str = ""
    args: list[CallArg] = field(default_factory=list)


@dataclass(slots=True)
class CastExpr(Node):
    expr: Expr | None = None
    type_name: str = ""


@dataclass(slots=True)
class CaseWhen(Node):
    condition: Expr | None = None
    result: Expr | None = None


@dataclass(slots=True)
class CaseExpr(Node):
    operand: Expr | None = None  # simple CASE
    whens: list[CaseWhen] = field(default_factory=list)
    else_result: Expr | None = None


Expr = (
    Name
    | Param
    | Literal
    | ArrayLiteral
    | UnaryOp
    | BinaryOp
    | IsPredicate
    | InPredicate
    | LikePredicate
    | BetweenPredicate
    | FuncCall
    | CastExpr
    | CaseExpr
)


@dataclass(slots=True)
class SelectItem(Node):
    star: bool = False
    expr: Expr | None = None
    alias: str | None = None


@dataclass(slots=True)
class OrderItem(Node):
    key: Name | FuncCall | None = None
    direction: str = "ASC"  # ASC|DESC
    nulls: str | None = None  # FIRST|LAST


@dataclass(slots=True)
class RetrieveCall(Node):
    """Canonical FROM retrieve(...) or legacy bare retriever call."""

    engine_name: str | None = None  # engine.<name>.retrieve
    calls: list[FuncCall] = field(default_factory=list)
    bare: bool = False  # legacy bare retriever (not wrapped in retrieve)


@dataclass(slots=True)
class TableRef(Node):
    path: list[str] = field(default_factory=list)


FromSource = RetrieveCall | TableRef


@dataclass(slots=True)
class SelectStmt(Node):
    select_list: list[SelectItem] = field(default_factory=list)
    from_sources: list[FromSource] = field(default_factory=list)
    where: Expr | None = None
    order_by: list[OrderItem] = field(default_factory=list)
    reorder_by: list[FuncCall] = field(default_factory=list)
    limit: int | Param | None = None
    offset: int | Param | None = None
