"""Shared expression evaluator (postfilter, computed columns, score value_model)."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Any, Callable

from recql.errors import ExecuteError
from recql.execute.merge import Candidate
from recql.language import ast as A
from recql.language.parser import parse_expr


@dataclass
class EvalContext:
    candidate: Candidate
    params: dict[str, Any] = field(default_factory=dict)
    item: dict[str, Any] = field(default_factory=dict)
    user: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


_CMP = {
    "=": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def eval_expr_ast(expr: A.Expr | None, ctx: EvalContext) -> Any:
    if expr is None:
        return None
    if isinstance(expr, A.Literal):
        return expr.value
    if isinstance(expr, A.Param):
        return _resolve_param(expr.value, ctx.params)
    if isinstance(expr, A.Name):
        return _resolve_name(expr.value, ctx)
    if isinstance(expr, A.ArrayLiteral):
        return [eval_expr_ast(e, ctx) for e in expr.elements]
    if isinstance(expr, A.UnaryOp):
        v = eval_expr_ast(expr.operand, ctx)
        if expr.op == "NOT":
            return not bool(v)
        if expr.op == "-":
            return -v
        raise ExecuteError(f"unknown unary op {expr.op}")
    if isinstance(expr, A.BinaryOp):
        if expr.op == "OR":
            return bool(eval_expr_ast(expr.left, ctx)) or bool(
                eval_expr_ast(expr.right, ctx)
            )
        if expr.op == "AND":
            return bool(eval_expr_ast(expr.left, ctx)) and bool(
                eval_expr_ast(expr.right, ctx)
            )
        left = eval_expr_ast(expr.left, ctx)
        right = eval_expr_ast(expr.right, ctx)
        if expr.op in _CMP:
            return _CMP[expr.op](left, right)
        if expr.op == "+":
            return left + right
        if expr.op == "-":
            return left - right
        if expr.op == "*":
            return left * right
        if expr.op == "/":
            return left / right
        if expr.op == "%":
            return left % right
        if expr.op == "**":
            return left**right
        raise ExecuteError(f"unknown binary op {expr.op}")
    if isinstance(expr, A.IsPredicate):
        v = eval_expr_ast(expr.expr, ctx)
        if expr.target == "NULL":
            ok = v is None
        elif expr.target == "TRUE":
            ok = v is True
        else:
            ok = v is False
        return (not ok) if expr.negated else ok
    if isinstance(expr, A.InPredicate):
        v = eval_expr_ast(expr.expr, ctx)
        if expr.param is not None:
            bag = _resolve_param(expr.param.value, ctx.params)
            if not isinstance(bag, (list, tuple, set)):
                raise ExecuteError("IN $param must resolve to a list")
            vals = list(bag)
        else:
            vals = [eval_expr_ast(x, ctx) for x in (expr.values or [])]
        ok = v in vals
        return (not ok) if expr.negated else ok
    if isinstance(expr, A.LikePredicate):
        v = str(eval_expr_ast(expr.expr, ctx) or "")
        pat = str(eval_expr_ast(expr.pattern, ctx) or "")
        # SQL LIKE: % → .*, _ → .
        import re

        flags = re.IGNORECASE if expr.case_insensitive else 0
        rx = re.escape(pat).replace(r"\%", ".*").replace(r"\_", ".")
        ok = re.fullmatch(rx, v, flags=flags) is not None
        return (not ok) if expr.negated else ok
    if isinstance(expr, A.BetweenPredicate):
        v = eval_expr_ast(expr.expr, ctx)
        lo = eval_expr_ast(expr.low, ctx)
        hi = eval_expr_ast(expr.high, ctx)
        ok = lo <= v <= hi
        return (not ok) if expr.negated else ok
    if isinstance(expr, A.FuncCall):
        return _eval_call(expr, ctx)
    if isinstance(expr, A.CastExpr):
        v = eval_expr_ast(expr.expr, ctx)
        t = expr.type_name.lower()
        if t in ("int", "integer"):
            return int(v)
        if t in ("float", "double", "real"):
            return float(v)
        if t in ("str", "string", "text"):
            return str(v)
        return v
    if isinstance(expr, A.CaseExpr):
        if expr.operand is not None:
            opv = eval_expr_ast(expr.operand, ctx)
            for w in expr.whens:
                if eval_expr_ast(w.condition, ctx) == opv:
                    return eval_expr_ast(w.result, ctx)
        else:
            for w in expr.whens:
                if eval_expr_ast(w.condition, ctx):
                    return eval_expr_ast(w.result, ctx)
        if expr.else_result is not None:
            return eval_expr_ast(expr.else_result, ctx)
        return None
    raise ExecuteError(f"cannot evaluate {type(expr).__name__}")


def eval_expr_string(source: str, ctx: EvalContext) -> Any:
    return eval_expr_ast(parse_expr(source), ctx)


def _resolve_param(value: str, params: dict[str, Any]) -> Any:
    body = value[1:] if value.startswith("$") else value
    for prefix in ("parameter.", "param.", "params."):
        if body.lower().startswith(prefix):
            body = body[len(prefix) :]
            break
    if body in params:
        return params[body]
    short = body.split(".")[-1]
    return params.get(short, value)


def _resolve_name(name: str, ctx: EvalContext) -> Any:
    # retrieval.<bag> / retrieval.<bag>_rank
    if name.startswith("retrieval."):
        key = name[len("retrieval.") :]
        if key in ctx.candidate.retrieval:
            return ctx.candidate.retrieval[key]
        # bare bag score alias
        return ctx.candidate.retrieval.get(key)
    if name == "item":
        return ctx.item or ctx.candidate.attributes
    if name == "user":
        return ctx.user
    if name.startswith("item."):
        return _dig(ctx.item or ctx.candidate.attributes, name[5:])
    if name.startswith("user."):
        return _dig(ctx.user, name[5:])
    if name == "retrieval_score":
        return ctx.candidate.retrieval_score
    if name in ctx.candidate.attributes:
        return ctx.candidate.attributes[name]
    if name in ctx.item:
        return ctx.item[name]
    if name in ctx.extras:
        return ctx.extras[name]
    if name in ctx.params:
        return ctx.params[name]
    # unresolved identifiers soft-fail as None for arithmetic defaults later
    return None


def _dig(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _eval_call(call: A.FuncCall, ctx: EvalContext) -> Any:
    name = call.name.lower()
    args = [_call_arg_value(a, ctx) for a in call.args]
    named = {
        a.name.lower(): _call_arg_value(a, ctx)
        for a in call.args
        if a.name
    }

    if name in ("retrieval.get_score", "get_score"):
        bag = args[0] if args else named.get("name")
        default = args[1] if len(args) > 1 else named.get("default", 0.0)
        return ctx.candidate.get_score(str(bag), float(default) if default is not None else 0.0)

    if name in ("retrieval.get_rank", "get_rank"):
        bag = args[0] if args else named.get("name")
        default = args[1] if len(args) > 1 else named.get("default", 999)
        rank = ctx.candidate.get_rank(str(bag), None)
        if rank is None:
            return int(default) if default is not None else 999
        return int(rank)

    if name == "abs":
        return abs(args[0])
    if name == "coalesce":
        for a in args:
            if a is not None:
                return a
        return None
    if name == "cosine_similarity":
        a, b = args[0], args[1]
        return _cosine(a, b)
    if name == "dot":
        return _dot(args[0], args[1])
    if name == "haversine_distance":
        if len(args) < 4:
            raise ExecuteError("haversine_distance requires lat1, lon1, lat2, lon2")
        return _haversine(
            float(args[0]), float(args[1]), float(args[2]), float(args[3])
        )
    if name == "now_seconds":
        import time

        return float(time.time())
    if name == "text_encoding":
        # text_encoding(entity, embedding_ref=…) — vectors from ctx.extras or attrs
        emb_ref = named.get("embedding_ref") or (args[1] if len(args) > 1 else None)
        entity = args[0] if args else named.get("entity")
        return _lookup_encoding(entity, emb_ref, ctx)
    if name == "pooled_text_encoding":
        emb_ref = named.get("embedding_ref")
        pool_fn = named.get("pool_fn") or "mean"
        seq = args[0] if args else None
        return _pooled_encoding(seq, emb_ref, str(pool_fn), ctx)
    if name in ("ln", "log"):
        return math.log(float(args[0]))
    if name == "sqrt":
        return math.sqrt(float(args[0]))
    if name == "pow":
        return float(args[0]) ** float(args[1])
    if name == "exp":
        return math.exp(float(args[0]))
    if name == "round":
        if len(args) > 1:
            return round(float(args[0]), int(args[1]))
        return round(float(args[0]))
    if name == "max":
        return max(args)
    if name == "min":
        return min(args)
    if name in ("array_has", "array_contains"):
        seq, val = args[0], args[1]
        if seq is None:
            return False
        return val in list(seq)
    if name == "array_has_any":
        seq, vals = args[0], args[1]
        if seq is None or vals is None:
            return False
        s = set(seq)
        return any(v in s for v in list(vals))
    if name == "array_has_all":
        seq, vals = args[0], args[1]
        if seq is None or vals is None:
            return False
        s = set(seq)
        return all(v in s for v in list(vals))
    if name == "regexp_match":
        import re

        text = "" if args[0] is None else str(args[0])
        pat = str(args[1])
        return re.search(pat, text) is not None
    if name in ("colbert_v2", "cross_encoder"):
        from recql.rerank import score_rerank

        item = args[0] if args else named.get("item")
        query = args[1] if len(args) > 1 else named.get("query")
        return score_rerank(name, _item_text(item), str(query or ""))

    # Soft fallback: unknown function name → error with diagnostics
    raise ExecuteError(f"unknown function in expression: {call.name}")


def _text_jaccard(item: Any, query: Any) -> float:
    from recql.pyutils.jaccard import jaccard

    a = set(str(_item_text(item)).lower())
    b = set(str(query or "").lower())
    return jaccard(a, b)


def _item_text(item: Any) -> str:
    if item is None:
        return ""
    if isinstance(item, dict):
        for k in ("title", "name", "description", "text"):
            if k in item and item[k] is not None:
                return str(item[k])
        return str(item)
    return str(item)


def _call_arg_value(arg: A.CallArg, ctx: EvalContext) -> Any:
    return eval_expr_ast(arg.value, ctx)


def _dot(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    va = list(a)
    vb = list(b)
    if len(va) != len(vb) or not va:
        return 0.0
    return float(sum(x * y for x, y in zip(va, vb, strict=True)))


def _cosine(a: Any, b: Any) -> float:
    if a is None or b is None:
        return 0.0
    va = list(a)
    vb = list(b)
    if len(va) != len(vb) or not va:
        return 0.0
    dot = _dot(va, vb)
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _lookup_encoding(entity: Any, emb_ref: Any, ctx: EvalContext) -> Any:
    encodings = ctx.extras.get("encodings") or {}
    if isinstance(entity, str) and entity in ("item", "user"):
        key = f"{entity}:{emb_ref}"
        if key in encodings:
            return encodings[key]
        if entity == "item":
            return ctx.candidate.attributes.get(f"encoding:{emb_ref}") or ctx.item.get(
                f"encoding:{emb_ref}"
            )
        return ctx.user.get(f"encoding:{emb_ref}")
    if isinstance(entity, (list, tuple)):
        return list(entity)
    return entity


def _pooled_encoding(seq: Any, emb_ref: Any, pool_fn: str, ctx: EvalContext) -> Any:
    if seq is None:
        return None
    vectors = []
    if isinstance(seq, list):
        for item in seq:
            if isinstance(item, (list, tuple)):
                vectors.append(list(item))
            else:
                v = _lookup_encoding(item, emb_ref, ctx)
                if v is not None:
                    vectors.append(list(v))
    if not vectors:
        return None
    dims = len(vectors[0])
    if pool_fn == "max":
        return [max(v[i] for v in vectors) for i in range(dims)]
    # mean
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dims)]


class ExpressionScorer:
    """Score candidates by evaluating ScoreEnsemble.value_model."""

    async def score_many(
        self, plan: Any, candidates: list[Candidate], ctx: dict[str, Any]
    ) -> list[float]:
        from recql.plugins.base import Scorer  # noqa: F401 — typing only

        vm = getattr(plan, "value_model", None)
        if not vm:
            return [float(c.retrieval_score or 0.0) for c in candidates]
        params = ctx.get("params") or {}
        user = ctx.get("user") or {}
        out: list[float] = []
        for c in candidates:
            ectx = EvalContext(
                candidate=c,
                params=params,
                item=dict(c.attributes),
                user=user,
            )
            val = eval_expr_string(str(vm), ectx)
            out.append(float(val) if val is not None else 0.0)
        return out
