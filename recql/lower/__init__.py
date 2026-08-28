"""AST → RankQueryConfig lower (OpenAPI IR)."""

from __future__ import annotations

import re
from typing import Any

from recql.errors import SyntaxError_, ValidationError
from recql.language import ast as A
from recql.openapi_ir import convert_rank_query_config, rank_query_config_to_dict
from recql.openapi_ir.models import RankQueryConfig


def normalize_param(value: str) -> str:
    """O-R2: $param.x / $params.x / $parameter.x → $parameter.x."""
    if not value.startswith("$"):
        return value
    body = value[1:]
    for prefix in ("parameter.", "param.", "params."):
        if body.lower().startswith(prefix):
            return "$parameter." + body[len(prefix) :]
    return value


def expr_to_string(expr: A.Expr | None) -> str:
    """Serialize an AST expr to a string (for where=/filter expression fields)."""
    if expr is None:
        return ""
    if isinstance(expr, A.Literal):
        if expr.kind == "string":
            return "'" + str(expr.value).replace("'", "''") + "'"
        if expr.kind == "null":
            return "NULL"
        if expr.kind == "bool":
            return "true" if expr.value else "false"
        return str(expr.value)
    if isinstance(expr, A.Name):
        return expr.value
    if isinstance(expr, A.Param):
        return normalize_param(expr.value)
    if isinstance(expr, A.ArrayLiteral):
        return "[" + ", ".join(expr_to_string(e) for e in expr.elements) + "]"
    if isinstance(expr, A.UnaryOp):
        if expr.op == "NOT":
            return f"NOT ({expr_to_string(expr.operand)})"
        return f"{expr.op}{expr_to_string(expr.operand)}"
    if isinstance(expr, A.BinaryOp):
        return f"({expr_to_string(expr.left)} {expr.op} {expr_to_string(expr.right)})"
    if isinstance(expr, A.IsPredicate):
        n = " NOT" if expr.negated else ""
        return f"({expr_to_string(expr.expr)} IS{n} {expr.target})"
    if isinstance(expr, A.InPredicate):
        n = "NOT " if expr.negated else ""
        if expr.param is not None:
            return f"({expr_to_string(expr.expr)} {n}IN {normalize_param(expr.param.value)})"
        vals = ", ".join(expr_to_string(v) for v in (expr.values or []))
        return f"({expr_to_string(expr.expr)} {n}IN ({vals}))"
    if isinstance(expr, A.LikePredicate):
        n = "NOT " if expr.negated else ""
        op = "ILIKE" if expr.case_insensitive else "LIKE"
        return f"({expr_to_string(expr.expr)} {n}{op} {expr_to_string(expr.pattern)})"
    if isinstance(expr, A.BetweenPredicate):
        n = "NOT " if expr.negated else ""
        return (
            f"({expr_to_string(expr.expr)} {n}BETWEEN {expr_to_string(expr.low)}"
            f" AND {expr_to_string(expr.high)})"
        )
    if isinstance(expr, A.FuncCall):
        return _call_to_string(expr)
    if isinstance(expr, A.CastExpr):
        return f"CAST({expr_to_string(expr.expr)} AS {expr.type_name})"
    if isinstance(expr, A.CaseExpr):
        parts = ["CASE"]
        if expr.operand is not None:
            parts.append(expr_to_string(expr.operand))
        for w in expr.whens:
            parts.append(
                f"WHEN {expr_to_string(w.condition)} THEN {expr_to_string(w.result)}"
            )
        if expr.else_result is not None:
            parts.append(f"ELSE {expr_to_string(expr.else_result)}")
        parts.append("END")
        return " ".join(parts)
    raise ValidationError(f"cannot stringify expr type {type(expr).__name__}")


def _call_to_string(call: A.FuncCall) -> str:
    args: list[str] = []
    for a in call.args:
        if a.name:
            args.append(f"{a.name}={expr_to_string(a.value)}")
        else:
            args.append(expr_to_string(a.value))
    return f"{call.name}({', '.join(args)})"


def _arg_map(call: A.FuncCall) -> dict[str, A.Expr]:
    out: dict[str, A.Expr] = {}
    for i, a in enumerate(call.args):
        if a.name:
            out[a.name.lower()] = a.value  # type: ignore[assignment]
        else:
            out[f"__pos_{i}"] = a.value  # type: ignore[assignment]
    return out


def _lit_or_param(expr: A.Expr | None) -> Any:
    if expr is None:
        return None
    if isinstance(expr, A.Literal):
        return expr.value
    if isinstance(expr, A.Param):
        return normalize_param(expr.value)
    if isinstance(expr, A.Name):
        return expr.value
    if isinstance(expr, A.ArrayLiteral):
        return [_lit_or_param(e) for e in expr.elements]
    return expr_to_string(expr)


def _parse_columns_dsl(text: str) -> list[dict[str, Any]]:
    """O-R4: 'price DESC, created_at ASC NULLS LAST' → ColumnOrdering dicts."""
    cols: list[dict[str, Any]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split()
        name = tokens[0]
        ascending = True
        nulls_first = False
        i = 1
        if i < len(tokens) and tokens[i].upper() in ("ASC", "DESC"):
            ascending = tokens[i].upper() == "ASC"
            i += 1
        if i < len(tokens) and tokens[i].upper() == "NULLS":
            i += 1
            if i < len(tokens) and tokens[i].upper() == "FIRST":
                nulls_first = True
            elif i < len(tokens) and tokens[i].upper() == "LAST":
                nulls_first = False
        cols.append({"name": name, "ascending": ascending, "nulls_first": nulls_first})
    return cols


def _lower_where_arg(expr: A.Expr | None) -> str | None:
    if expr is None:
        return None
    # LEGACY: string where= re-parse would happen at annotate; here stringify or use as-is
    if isinstance(expr, A.Literal) and expr.kind == "string":
        return str(expr.value)
    return expr_to_string(expr)


def _encoder_user_id(emap: dict[str, A.Expr], args: dict[str, A.Expr]) -> Any:
    uid = (
        emap.get("input_user_id")
        or emap.get("__pos_0")
        or args.get("input_user_id")
        or A.Param(value="$parameter.user_id")
    )
    return _normalize_id_ref(_lit_or_param(uid))


def _normalize_id_ref(val: Any) -> Any:
    """Docs often write input_user_id='$user_id' as a string literal."""
    if isinstance(val, str) and val.startswith("$"):
        return normalize_param(val)
    return val


def _lower_encoder(expr: A.Expr | None, args: dict[str, A.Expr]) -> dict[str, Any]:
    """O-R3: always emit object-form query_encoder."""
    if isinstance(expr, A.FuncCall):
        t = expr.name.lower()
        emap = _arg_map(expr)
        return _encoder_object(t, emap, args)
    if isinstance(expr, A.Name) or isinstance(expr, A.Literal):
        t = str(expr.value if isinstance(expr, A.Literal) else expr.value).lower()
        return _encoder_object(t, {}, args)
    raise ValidationError(f"unsupported encoder form: {expr!r}")


def _encoder_object(
    t: str, emap: dict[str, A.Expr], args: dict[str, A.Expr]
) -> dict[str, Any]:
    if t == "precomputed_user":
        return {
            "type": "precomputed_user",
            "input_user_id": _encoder_user_id(emap, args),
        }
    if t == "precomputed_item":
        iid = (
            emap.get("input_item_id")
            or emap.get("__pos_0")
            or args.get("input_item_id")
        )
        return {
            "type": "precomputed_item",
            "input_item_id": _normalize_id_ref(_lit_or_param(iid)),
        }
    if t == "interaction_pooling":
        trunc = _lit_or_param(
            emap.get("truncate_interactions") or args.get("truncate_interactions")
        )
        return {
            "type": "interaction_pooling",
            "input_user_id": _encoder_user_id(emap, args),
            "pooling_function": _lit_or_param(emap.get("pooling_function")) or "mean",
            "truncate_interactions": int(trunc) if trunc is not None else 10,
        }
    if t == "interaction_round_robin":
        n = _lit_or_param(emap.get("num_clusters") or args.get("num_clusters"))
        return {
            "type": "interaction_round_robin",
            "input_user_id": _encoder_user_id(emap, args),
            "pooling_function": _lit_or_param(emap.get("pooling_function")) or "mean",
            "num_clusters": int(n) if n is not None else 5,
        }
    if t == "user_attribute_pooling":
        return {
            "type": "user_attribute_pooling",
            "input_user_id": _encoder_user_id(emap, args),
            "input_user_features": _lit_or_param(
                emap.get("input_user_features") or args.get("input_user_features")
            ),
        }
    if t == "item_attribute_pooling":
        return {
            "type": "item_attribute_pooling",
            "input_item_id": _normalize_id_ref(
                _lit_or_param(
                    emap.get("input_item_id") or args.get("input_item_id")
                )
            ),
            "input_item_features": _lit_or_param(
                emap.get("input_item_features") or args.get("input_item_features")
            ),
        }
    return {"type": t}


def _lower_retrieve_call(call: A.FuncCall) -> dict[str, Any]:
    name = call.name.lower()
    # Map SQL names → OpenAPI type
    type_map = {
        "similarity": "similarity",
        "similar_items": "similarity",
        "text_search": "text_search",
        "column_order": "column_order",
        "filter": "filter",
        "ids": "candidate_ids",
        "candidate_ids": "candidate_ids",
        "candidate_attributes": "candidate_attributes",
    }
    rtype = type_map.get(name)
    if rtype is None:
        raise ValidationError(f"unknown retriever: {call.name}")

    args = _arg_map(call)
    step: dict[str, Any] = {"type": rtype}

    if "name" in args:
        step["name"] = _lit_or_param(args["name"])
    if "limit" in args:
        lim = _lit_or_param(args["limit"])
        # O-R6 extended: param retrieve limits deferred (omit → OpenAPI default).
        if not (isinstance(lim, str) and lim.startswith("$")) and lim is not None:
            step["limit"] = int(lim)
    if "where" in args:
        step["where"] = _lower_where_arg(args["where"])

    if rtype == "similarity":
        emb = args.get("embedding_ref") or args.get("embedding")
        if emb is None:
            raise ValidationError("similarity requires embedding_ref")
        step["embedding_ref"] = _lit_or_param(emb)
        enc = args.get("query_encoder") or args.get("encoder")
        if enc is None:
            # infer from input_* args (docs often omit encoder= when inputs imply it)
            if "input_item_id" in args:
                enc = A.Name(value="precomputed_item")
            elif (
                "input_user_id" in args
                or "input_interactions_item_ids" in args
                or "input_user_features" in args
            ):
                enc = A.Name(value="precomputed_user")
            else:
                # O-R10: docs often show bare similarity(embedding_ref=…);
                # default to precomputed_user + $parameter.user_id for OpenAPI.
                enc = A.Name(value="precomputed_user")
                if "input_user_id" not in args:
                    args = {
                        **args,
                        "input_user_id": A.Param(value="$parameter.user_id"),
                    }
        step["query_encoder"] = _lower_encoder(enc, args)
        if "use_exact_search" in args:
            step["use_exact_search"] = bool(_lit_or_param(args["use_exact_search"]))

    elif rtype == "text_search":
        q = args.get("input_text_query") or args.get("query")
        if q is None:
            raise ValidationError("text_search requires query / input_text_query")
        step["input_text_query"] = _lit_or_param(q)
        mode = args.get("mode")
        if mode is None:
            # O-R11: docs sometimes omit mode; default lexical when no embedding ref,
            # else vector when text_embedding_ref / embedding_ref present.
            if args.get("text_embedding_ref") or args.get("embedding_ref"):
                mode = A.Literal(value="vector", kind="string")
            else:
                mode = A.Literal(value="lexical", kind="string")
        mode_val = _lit_or_param(mode)
        if isinstance(mode_val, str) and mode_val.lower() == "lexical":
            fuzz = args.get("fuzziness") or args.get("fuzziness_edit_distance")
            mode_obj: dict[str, Any] = {"type": "lexical"}
            if fuzz is not None:
                mode_obj["fuzziness_edit_distance"] = int(_lit_or_param(fuzz) or 0)
            step["mode"] = mode_obj
        elif isinstance(mode_val, str) and mode_val.lower() == "vector":
            ref = (
                args.get("text_embedding_ref")
                or args.get("embedding_ref")
                or args.get("text_embedding")
            )
            # O-R13: docs often write mode='vector' without ref; default content_embedding.
            step["mode"] = {
                "type": "vector",
                "text_embedding_ref": _lit_or_param(ref)
                if ref is not None
                else "content_embedding",
            }
        elif isinstance(mode, A.FuncCall):
            mn = mode.name.lower()
            mmap = _arg_map(mode)
            if mn == "lexical":
                step["mode"] = {
                    "type": "lexical",
                    "fuzziness_edit_distance": int(
                        _lit_or_param(mmap.get("fuzziness_edit_distance")) or 0
                    ),
                }
            elif mn == "vector":
                step["mode"] = {
                    "type": "vector",
                    "text_embedding_ref": _lit_or_param(
                        mmap.get("text_embedding_ref")
                        or args.get("text_embedding_ref")
                    ),
                }
            else:
                raise ValidationError(f"unknown text_search mode: {mn}")
        else:
            raise ValidationError(f"unsupported text_search mode: {mode_val!r}")

    elif rtype == "column_order":
        cols = args.get("columns")
        if cols is None:
            raise ValidationError("column_order requires columns")
        if isinstance(cols, A.Literal) and cols.kind == "string":
            step["columns"] = _parse_columns_dsl(str(cols.value))
        elif isinstance(cols, A.ArrayLiteral):
            step["columns"] = []
            for el in cols.elements:
                if isinstance(el, A.Literal) and el.kind == "string":
                    step["columns"].extend(_parse_columns_dsl(str(el.value)))
                else:
                    step["columns"].append(
                        {"name": _lit_or_param(el), "ascending": True, "nulls_first": False}
                    )
        else:
            step["columns"] = _parse_columns_dsl(expr_to_string(cols))

    elif rtype == "filter":
        pass  # where/limit already set

    elif rtype == "candidate_ids":
        ids = args.get("item_ids") or args.get("ids") or args.get("__pos_0")
        if ids is None:
            raise ValidationError("candidate_ids requires item_ids / ids")
        step["item_ids"] = _lit_or_param(ids)

    elif rtype == "candidate_attributes":
        attrs = args.get("item_attributes") or args.get("__pos_0")
        if attrs is None:
            raise ValidationError("candidate_attributes requires item_attributes")
        step["item_attributes"] = _lit_or_param(attrs)

    return step


def _lower_score_call(call: A.FuncCall, *, alias: str | None, preserve_order: bool) -> dict[str, Any]:
    args = _arg_map(call)
    value_model = args.get("expression") or args.get("value_model") or args.get("__pos_0")
    if value_model is None:
        raise ValidationError("score() requires expression / value_model")
    if isinstance(value_model, A.Literal) and value_model.kind == "string":
        vm = str(value_model.value)
    elif isinstance(value_model, A.Name):
        vm = value_model.value
    else:
        vm = expr_to_string(value_model)
    out: dict[str, Any] = {
        "type": "score_ensemble",
        "value_model": vm,
        "preserve_order": preserve_order,
    }
    if alias:
        out["output_alias"] = alias
    if "name" in args:
        out["name"] = _lit_or_param(args["name"])
    if "input_user_id" in args:
        out["input_user_id"] = _lit_or_param(args["input_user_id"])
    if "input_user_features" in args:
        out["input_user_features"] = _lit_or_param(args["input_user_features"])
    if "input_interactions_item_ids" in args:
        out["input_interactions_item_ids"] = _lit_or_param(
            args["input_interactions_item_ids"]
        )
    return out


def _lit_or_param_number(expr: A.Expr | None, *, default: float | None = None) -> Any:
    """Return float/int literal, leave $param as string, else default."""
    val = _lit_or_param(expr)
    if val is None:
        return default
    if isinstance(val, str) and val.startswith("$"):
        return val
    try:
        return float(val)
    except (TypeError, ValueError):
        return default if default is not None else val


def _lower_reorder_call(call: A.FuncCall, *, alias: str | None = None) -> dict[str, Any]:
    name = call.name.lower()
    args = _arg_map(call)
    if name == "diversity":
        step: dict[str, Any] = {"type": "diversity"}
        if "strength" in args:
            step["strength"] = _lit_or_param_number(args["strength"], default=0.5)
        if "text_encoding_embedding_ref" in args:
            step["text_encoding_embedding_ref"] = _lit_or_param(
                args["text_encoding_embedding_ref"]
            )
        if "diversity_attributes" in args:
            step["diversity_attributes"] = _lit_or_param(args["diversity_attributes"])
        if alias:
            step["output_alias"] = alias
        if "name" in args:
            step["name"] = _lit_or_param(args["name"])
        return step
    if name in ("exploration", "boosted"):
        ret = args.get("retriever")
        step = {
            "type": name,
            "strength": _lit_or_param_number(args.get("strength"), default=0.5),
        }
        if ret is None:
            # O-R12: docs shorthand exploration(strength=0.2) without inline retriever —
            # emit a placeholder column_order cold-start bag for structural validity.
            step["retriever"] = {
                "type": "column_order",
                "columns": [{"name": "_derived_popular_rank", "ascending": True}],
                "limit": 50,
                "name": f"{name}_pool",
            }
        elif isinstance(ret, A.FuncCall):
            step["retriever"] = _lower_retrieve_call(ret)
        elif isinstance(ret, A.Name) or isinstance(ret, A.Literal):
            # named bag reference — not a full retrieve step; placeholder
            step["retriever"] = {
                "type": "column_order",
                "columns": [{"name": "_derived_popular_rank", "ascending": True}],
                "limit": 50,
                "name": str(_lit_or_param(ret)),
            }
        else:
            raise ValidationError(f"{name}() retriever must be a retrieve call")
        if alias:
            step["output_alias"] = alias
        if "name" in args:
            step["name"] = _lit_or_param(args["name"])
        return step
    raise ValidationError(f"unknown reorder function: {call.name}")


def lower_select(stmt: A.SelectStmt) -> RankQueryConfig:
    """Lower a parsed SelectStmt to RankQueryConfig and structurally validate."""
    retrieve: list[dict[str, Any]] = []
    entity_from: str | None = "item"

    for src in stmt.from_sources:
        if isinstance(src, A.RetrieveCall):
            for call in src.calls:
                retrieve.append(_lower_retrieve_call(call))
        elif isinstance(src, A.TableRef):
            # items/users bare tables — treat as filter retrieve of all
            if src.path and src.path[-1].lower() == "users":
                entity_from = "user"
            retrieve.append({"type": "filter", "limit": 1000})
        else:
            raise ValidationError(f"unsupported FROM source: {type(src).__name__}")

    filters: list[dict[str, Any]] | None = None
    if stmt.where is not None:
        # postfilter
        if isinstance(stmt.where, A.FuncCall) and stmt.where.name.lower() == "prebuilt":
            amap = _arg_map(stmt.where)
            ref = amap.get("__pos_0") or amap.get("filter_ref")
            fstep: dict[str, Any] = {
                "type": "prebuilt",
                "filter_ref": _lit_or_param(ref),
            }
            if "input_user_id" in amap:
                fstep["input_user_id"] = _lit_or_param(amap["input_user_id"])
            if "input_item_id" in amap:
                fstep["input_item_id"] = _lit_or_param(amap["input_item_id"])
            filters = [fstep]
        else:
            filters = [{"type": "expression", "expression": expr_to_string(stmt.where)}]

    score: dict[str, Any] | None = None
    reorder: list[dict[str, Any]] = []
    computed: list[dict[str, Any]] = []
    order_columns: list[dict[str, Any]] = []

    has_order_by = bool(stmt.order_by)

    reorder_aliases: set[str] = set()
    for item in stmt.select_list:
        if item.star:
            continue
        expr = item.expr
        if isinstance(expr, A.FuncCall):
            fname = expr.name.lower()
            if fname == "score":
                # preserve_order when score in SELECT without ORDER BY
                preserve = not has_order_by
                # unless ORDER BY references this alias
                if has_order_by and item.alias:
                    for oi in stmt.order_by:
                        if isinstance(oi.key, A.Name) and oi.key.value == item.alias:
                            preserve = False
                            break
                score = _lower_score_call(expr, alias=item.alias, preserve_order=preserve)
            elif fname in ("diversity", "exploration", "boosted"):
                reorder.append(_lower_reorder_call(expr, alias=item.alias))
                if item.alias:
                    reorder_aliases.add(str(item.alias))
            else:
                # computed column
                if item.alias is None:
                    raise SyntaxError_(
                        "computed columns require AS alias",
                        locations=[],
                    )
                computed.append(
                    {
                        "type": "computed_column",
                        "value_model": expr_to_string(expr),
                        "output_alias": item.alias,
                    }
                )
        elif item.alias is not None:
            computed.append(
                {
                    "type": "computed_column",
                    "value_model": expr_to_string(expr),
                    "output_alias": item.alias,
                }
            )

    # ORDER BY score(...) legacy → score step
    score_alias = (score or {}).get("output_alias") if score else None
    for oi in stmt.order_by:
        if isinstance(oi.key, A.FuncCall) and oi.key.name.lower() == "score":
            if score is None:
                score = _lower_score_call(oi.key, alias=None, preserve_order=False)
            elif score.get("preserve_order"):
                score["preserve_order"] = False
            # Sorting is owned by the score stage — do not emit column_sort.
        elif isinstance(oi.key, A.Name):
            # ORDER BY <score alias> → score stage sorts; skip column_sort for that key.
            if score_alias and oi.key.value == score_alias:
                if score is not None:
                    score["preserve_order"] = False
                continue
            # ORDER BY <reorder alias> → keep diversity/exploration/boosted order.
            if oi.key.value in reorder_aliases:
                continue
            order_columns.append(
                {
                    "name": oi.key.value,
                    "ascending": oi.direction == "ASC",
                    "nulls_first": oi.nulls == "FIRST",
                }
            )
        elif isinstance(oi.key, A.FuncCall):
            # ORDER BY diversity(...) etc. — treat as reorder
            reorder.append(_lower_reorder_call(oi.key))

    for call in stmt.reorder_by:
        reorder.append(_lower_reorder_call(call))

    if order_columns:
        reorder.append({"type": "column_sort", "columns": order_columns})

    limit = None
    if isinstance(stmt.limit, int):
        limit = stmt.limit
    elif isinstance(stmt.limit, A.Param):
        # keep as-is via string in a side channel — OpenAPI limit is int|null;
        # param limits resolved at bind. Store sentinel string in extensions via
        # leaving limit None and encoding in computed — for Phase A, stringify param
        # into a special form is wrong. Resolve: leave None and attach later.
        # O-R6: param LIMIT deferred to bind; store as None here and pass via
        # unbound params. For roundtrip goldens, use int limits.
        limit = None

    offset = None
    if isinstance(stmt.offset, int):
        offset = stmt.offset

    raw: dict[str, Any] = {
        "type": "rank",
        "from": entity_from,
        "retrieve": retrieve,
        "limit": limit,
        "offset": offset,
    }
    if filters is not None:
        raw["filter"] = filters
    if computed:
        raw["computed_columns"] = computed
    if score is not None:
        raw["score"] = score
    if reorder:
        raw["reorder"] = reorder

    return convert_rank_query_config(raw)


def lower_to_dict(stmt: A.SelectStmt) -> dict[str, Any]:
    return rank_query_config_to_dict(lower_select(stmt))


_PARAM_LIMIT = re.compile(r"\$parameter\.")
