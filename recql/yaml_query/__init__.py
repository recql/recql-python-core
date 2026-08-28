"""Load / validate / normalize QueryDefinition and RankQueryConfig."""

from __future__ import annotations

import re
from typing import Any

import msgspec
import yaml

from recql.errors import ValidationError
from recql.language.parser import parse
from recql.lower import lower_select, normalize_param
from recql.openapi_ir import (
    QueryDefinition,
    RankQueryConfig,
    convert_query_definition,
    convert_rank_query_config,
    rank_query_config_to_dict,
)


def load_yaml(text: str) -> Any:
    return yaml.safe_load(text)


def load_rank_query_config(obj: Any) -> RankQueryConfig:
    """Load dict/YAML/JSON object as RankQueryConfig."""
    if isinstance(obj, str):
        obj = load_yaml(obj)
    try:
        return convert_rank_query_config(obj)
    except (TypeError, msgspec.ValidationError) as e:
        raise ValidationError(str(e)) from e


def load_query_definition(obj: Any) -> QueryDefinition:
    if isinstance(obj, str):
        obj = load_yaml(obj)
    try:
        return convert_query_definition(obj)
    except (TypeError, msgspec.ValidationError) as e:
        raise ValidationError(str(e)) from e


def _looks_like_structured_query(text: str) -> bool:
    """True if text is YAML/JSON RankQueryConfig / QueryDefinition, not RecQL SQL."""
    s = text.strip()
    if s.startswith("{") or s.startswith("["):
        return True
    # Strip leading YAML comments
    lines = [ln for ln in s.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if not lines:
        return False
    head = "\n".join(lines[:8])
    return bool(
        re.search(r"(?m)^(type:\s*rank\b|query:\s*$|retrieve:\s*$|from:\s*)", head)
    )


def query_input_to_rank_query_config(query: Any) -> RankQueryConfig:
    """Accept RecQL string | RankQueryConfig | QueryDefinition | dict → RankQueryConfig."""
    if isinstance(query, RankQueryConfig):
        return query
    if isinstance(query, QueryDefinition):
        q = query.query
        if isinstance(q, str):
            return lower_select(parse(q))
        if isinstance(q, RankQueryConfig):
            return q
        return convert_rank_query_config(q)
    if isinstance(query, str):
        if _looks_like_structured_query(query):
            loaded = load_yaml(query)
            return query_input_to_rank_query_config(loaded)
        return lower_select(parse(query))
    if isinstance(query, dict):
        if "query" in query and (
            isinstance(query["query"], (str, dict))
            or "parameters" in query
        ) and "retrieve" not in query:
            # QueryDefinition-shaped
            qd = load_query_definition(query)
            return query_input_to_rank_query_config(qd)
        if query.get("type") == "rank" or "retrieve" in query:
            return load_rank_query_config(query)
        if "query" in query:
            qd = load_query_definition(query)
            return query_input_to_rank_query_config(qd)
        raise ValidationError("unrecognized query dict shape")
    raise ValidationError(f"unsupported query type: {type(query).__name__}")


def normalize_config_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Normalize for ROUNDTRIP comparison (O-R2/O-R3/O-R4)."""

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k in (
                    "input_user_id",
                    "input_item_id",
                    "input_text_query",
                    "filter_ref",
                    "embedding_ref",
                    "text_embedding_ref",
                ) and isinstance(v, str):
                    out[k] = normalize_param(v)
                else:
                    out[k] = walk(v)
            # drop null optionals for stable compare
            return {k: v for k, v in out.items() if v is not None and v != [] and v != {}}
        if isinstance(obj, list):
            return [walk(x) for x in obj]
        if isinstance(obj, str) and obj.startswith("$"):
            return normalize_param(obj)
        return obj

    return walk(rank_query_config_to_dict(convert_rank_query_config(d)) if "type" in d or "retrieve" in d else walk(d))
