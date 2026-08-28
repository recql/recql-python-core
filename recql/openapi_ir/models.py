"""OpenAPI-aligned msgspec IR for QueryDefinition / RankQueryConfig.

Source of truth: side-engine/api/openapi.yaml components.schemas
(QueryDefinition, RankQueryConfig, nested retrieve/filter/score/reorder/
encoder steps). Hand-maintained mirror (O-R1).
"""

from __future__ import annotations

from typing import Any, Literal, Union

import msgspec

# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class PrecomputedUserEmbedding(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    input_user_id: str
    type: Literal["precomputed_user"] = "precomputed_user"


class PrecomputedItemEmbedding(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    input_item_id: str
    type: Literal["precomputed_item"] = "precomputed_item"


class InteractionPoolingEncoder(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    input_user_id: str
    pooling_function: str = "mean"
    truncate_interactions: int = 10
    type: Literal["interaction_pooling"] = "interaction_pooling"


class InteractionRoundRobinEncoder(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    input_user_id: str
    pooling_function: str = "mean"
    num_clusters: int = 5
    type: Literal["interaction_round_robin"] = "interaction_round_robin"


class UserAttributePoolingEncoder(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    input_user_id: str | None = None
    # Docs pass '$user_features' param refs as well as inline objects.
    input_user_features: Any = None
    type: Literal["user_attribute_pooling"] = "user_attribute_pooling"


class ItemAttributePoolingEncoder(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    input_item_id: str | None = None
    input_item_features: Any = None
    type: Literal["item_attribute_pooling"] = "item_attribute_pooling"


QueryEncoder = Union[
    InteractionPoolingEncoder,
    InteractionRoundRobinEncoder,
    UserAttributePoolingEncoder,
    PrecomputedUserEmbedding,
    ItemAttributePoolingEncoder,
    PrecomputedItemEmbedding,
]

_ENCODER_BY_TYPE: dict[str, type] = {
    "precomputed_user": PrecomputedUserEmbedding,
    "precomputed_item": PrecomputedItemEmbedding,
    "interaction_pooling": InteractionPoolingEncoder,
    "interaction_round_robin": InteractionRoundRobinEncoder,
    "user_attribute_pooling": UserAttributePoolingEncoder,
    "item_attribute_pooling": ItemAttributePoolingEncoder,
}


class LexicalSearchMode(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    fuzziness_edit_distance: int = 0
    type: Literal["lexical"] = "lexical"


class VectorSearchMode(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    text_embedding_ref: str
    type: Literal["vector"] = "vector"
    use_exact_search: bool = False


SearchMode = Union[LexicalSearchMode, VectorSearchMode]


class ColumnOrdering(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str
    ascending: bool = True
    nulls_first: bool = False


class ColumnOrderRetrieveStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    columns: list[ColumnOrdering]
    where: str | None = None
    limit: int = 100
    name: str | None = None
    type: Literal["column_order"] = "column_order"


class TextSearchRetrieveStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    input_text_query: str
    mode: Any  # SearchMode — converted manually
    where: str | None = None
    limit: int = 100
    name: str | None = None
    type: Literal["text_search"] = "text_search"


class SimilarityRetrieveStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    embedding_ref: str
    query_encoder: Any  # QueryEncoder
    where: str | None = None
    limit: int = 100
    name: str | None = None
    type: Literal["similarity"] = "similarity"
    use_exact_search: bool = False


class FilterRetrieveStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    where: str | None = None
    limit: int = 100
    name: str | None = None
    type: Literal["filter"] = "filter"


class CandidateIdsRetrieveStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    # OpenAPI: array; RecQL also allows a parameter ref string resolved at bind/execute.
    item_ids: Any
    limit: int | None = None
    name: str | None = None
    type: Literal["candidate_ids"] = "candidate_ids"


class CandidateAttributesRetrieveStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    item_attributes: Any  # list[dict] or parameter ref string
    limit: int | None = None
    name: str | None = None
    type: Literal["candidate_attributes"] = "candidate_attributes"


RetrieveStep = Union[
    ColumnOrderRetrieveStep,
    TextSearchRetrieveStep,
    SimilarityRetrieveStep,
    FilterRetrieveStep,
    CandidateIdsRetrieveStep,
    CandidateAttributesRetrieveStep,
]

_RETRIEVE_BY_TYPE: dict[str, type] = {
    "column_order": ColumnOrderRetrieveStep,
    "text_search": TextSearchRetrieveStep,
    "similarity": SimilarityRetrieveStep,
    "filter": FilterRetrieveStep,
    "candidate_ids": CandidateIdsRetrieveStep,
    "candidate_attributes": CandidateAttributesRetrieveStep,
}


class PrebuiltFilterStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    filter_ref: str
    name: str | None = None
    input_user_id: str | None = None
    input_item_id: str | None = None
    type: Literal["prebuilt"] = "prebuilt"


class ExpressionFilterStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    expression: str
    name: str | None = None
    type: Literal["expression"] = "expression"


class TruncateFilterStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str | None = None
    max_length: int = 500
    type: Literal["truncate"] = "truncate"


FilterStep = Union[PrebuiltFilterStep, ExpressionFilterStep, TruncateFilterStep]

_FILTER_BY_TYPE: dict[str, type] = {
    "prebuilt": PrebuiltFilterStep,
    "expression": ExpressionFilterStep,
    "truncate": TruncateFilterStep,
}


class ScoreEnsemble(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    value_model: str
    input_user_id: str | None = None
    input_user_features: dict[str, Any] | None = None
    # OpenAPI: array; docs often pass a parameter ref string.
    input_interactions_item_ids: Any = None
    name: str | None = None
    output_alias: str | None = None
    preserve_order: bool = False
    type: Literal["score_ensemble"] = "score_ensemble"


class PassthroughScore(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    name: str | None = None
    type: Literal["passthrough"] = "passthrough"


ScoreStep = Union[ScoreEnsemble, PassthroughScore]


class ExplorationReorderStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    retriever: Any
    strength: Any = 0.5
    name: str | None = None
    output_alias: str | None = None
    type: Literal["exploration"] = "exploration"


class BoostedReorderStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    retriever: Any
    strength: Any = 0.5
    name: str | None = None
    output_alias: str | None = None
    type: Literal["boosted"] = "boosted"


class DiversityReorderStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    strength: Any = 0.5
    diversity_lookback_window: int = 30
    diversity_lookforward_window: int = 30
    max_diversity_candidates: int = 1000
    diversity_attributes: list[Any] | None = None
    text_encoding_embedding_ref: str | None = None
    name: str | None = None
    output_alias: str | None = None
    type: Literal["diversity"] = "diversity"


class ColumnSortReorderStep(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    columns: list[ColumnOrdering]
    name: str | None = None
    output_alias: str | None = None
    type: Literal["column_sort"] = "column_sort"


ReorderStep = Union[
    ExplorationReorderStep,
    BoostedReorderStep,
    DiversityReorderStep,
    ColumnSortReorderStep,
]


class ComputedColumn(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    value_model: str
    output_alias: str
    input_user_id: str | None = None
    input_user_features: dict[str, Any] | None = None
    input_interactions_item_ids: Any = None
    name: str | None = None
    preserve_order: bool = False
    type: Literal["computed_column"] = "computed_column"


class ParameterDefinition(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    type: str | None = None
    default: Any = None


class RankQueryConfig(msgspec.Struct, frozen=True, forbid_unknown_fields=True, rename={"from_": "from"}):
    """Executable IR — OpenAPI RankQueryConfig. Field ``from_`` maps to JSON ``from``."""

    retrieve: list[Any] = []
    filter: list[Any] | None = None
    computed_columns: list[Any] = []
    score: Any = None
    reorder: list[Any] = []
    limit: int | None = None
    offset: int | None = None
    columns: list[str] | None = None
    embeddings: list[str] | None = None
    type: Literal["rank"] = "rank"
    from_: Literal["user", "item", "item_attribute"] | None = None


class QueryDefinition(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    query: Any  # RankQueryConfig | str
    parameters: dict[str, ParameterDefinition] | None = None


def _convert_encoder(obj: Any) -> QueryEncoder:
    if not isinstance(obj, dict):
        raise TypeError(f"encoder must be object, got {type(obj).__name__}")
    t = obj.get("type")
    cls = _ENCODER_BY_TYPE.get(t)  # type: ignore[arg-type]
    if cls is None:
        raise TypeError(f"unknown query_encoder type: {t!r}")
    return msgspec.convert(obj, type=cls)


def _convert_mode(obj: Any) -> SearchMode:
    if isinstance(obj, str):
        if obj == "lexical":
            return LexicalSearchMode()
        raise TypeError(f"search mode string {obj!r} requires object form for vector")
    if not isinstance(obj, dict):
        raise TypeError(f"mode must be object, got {type(obj).__name__}")
    t = obj.get("type")
    if t == "lexical":
        return msgspec.convert(obj, type=LexicalSearchMode)
    if t == "vector":
        return msgspec.convert(obj, type=VectorSearchMode)
    raise TypeError(f"unknown search mode type: {t!r}")


def _convert_retrieve(obj: Any) -> RetrieveStep:
    if not isinstance(obj, dict):
        raise TypeError(f"retrieve step must be object, got {type(obj).__name__}")
    t = obj.get("type")
    if t == "similarity":
        data = dict(obj)
        if "query_encoder" in data:
            data["query_encoder"] = _convert_encoder(data["query_encoder"])
        return msgspec.convert(data, type=SimilarityRetrieveStep)
    if t == "text_search":
        data = dict(obj)
        if "mode" in data:
            data["mode"] = _convert_mode(data["mode"])
        return msgspec.convert(data, type=TextSearchRetrieveStep)
    if t == "column_order":
        data = dict(obj)
        if "columns" in data and isinstance(data["columns"], list):
            data["columns"] = [
                msgspec.convert(c, type=ColumnOrdering) if isinstance(c, dict) else c
                for c in data["columns"]
            ]
        return msgspec.convert(data, type=ColumnOrderRetrieveStep)
    cls = _RETRIEVE_BY_TYPE.get(t)  # type: ignore[arg-type]
    if cls is None:
        raise TypeError(f"unknown retrieve type: {t!r}")
    return msgspec.convert(obj, type=cls)


def _convert_filter(obj: Any) -> FilterStep:
    if isinstance(obj, str):
        return ExpressionFilterStep(expression=obj)
    if not isinstance(obj, dict):
        raise TypeError(f"filter step must be object, got {type(obj).__name__}")
    t = obj.get("type")
    if t is None and "expression" in obj:
        return msgspec.convert({**obj, "type": "expression"}, type=ExpressionFilterStep)
    if t is None and "filter_ref" in obj:
        return msgspec.convert({**obj, "type": "prebuilt"}, type=PrebuiltFilterStep)
    cls = _FILTER_BY_TYPE.get(t)  # type: ignore[arg-type]
    if cls is None:
        raise TypeError(f"unknown filter type: {t!r}")
    return msgspec.convert(obj, type=cls)


def _convert_score(obj: Any) -> ScoreStep | None:
    if obj is None:
        return None
    if not isinstance(obj, dict):
        raise TypeError(f"score must be object, got {type(obj).__name__}")
    t = obj.get("type", "score_ensemble")
    if t == "passthrough":
        return msgspec.convert(obj, type=PassthroughScore)
    if t == "score_ensemble" or "value_model" in obj:
        return msgspec.convert({**obj, "type": "score_ensemble"}, type=ScoreEnsemble)
    raise TypeError(f"unknown score type: {t!r}")


def _convert_reorder(obj: Any) -> ReorderStep:
    if not isinstance(obj, dict):
        raise TypeError(f"reorder step must be object, got {type(obj).__name__}")
    t = obj.get("type")
    if t in ("exploration", "boosted"):
        data = dict(obj)
        if "retriever" in data:
            data["retriever"] = _convert_retrieve(data["retriever"])
        cls = ExplorationReorderStep if t == "exploration" else BoostedReorderStep
        return msgspec.convert(data, type=cls)
    if t == "diversity":
        return msgspec.convert(obj, type=DiversityReorderStep)
    if t == "column_sort":
        data = dict(obj)
        if "columns" in data:
            data["columns"] = [
                msgspec.convert(c, type=ColumnOrdering) if isinstance(c, dict) else c
                for c in data["columns"]
            ]
        return msgspec.convert(data, type=ColumnSortReorderStep)
    raise TypeError(f"unknown reorder type: {t!r}")


def convert_rank_query_config(obj: Any) -> RankQueryConfig:
    """Validate/convert a dict to RankQueryConfig (OpenAPI structural validate)."""
    if isinstance(obj, RankQueryConfig):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"RankQueryConfig must be object, got {type(obj).__name__}")
    data = dict(obj)
    if data.get("type", "rank") != "rank":
        raise TypeError(f"expected type=rank, got {data.get('type')!r}")
    data["type"] = "rank"
    data["retrieve"] = [_convert_retrieve(s) for s in (data.get("retrieve") or [])]
    if data.get("filter") is not None:
        data["filter"] = [_convert_filter(s) for s in data["filter"]]
    data["computed_columns"] = [
        msgspec.convert(c, type=ComputedColumn) for c in (data.get("computed_columns") or [])
    ]
    if "score" in data:
        data["score"] = _convert_score(data["score"])
    data["reorder"] = [_convert_reorder(s) for s in (data.get("reorder") or [])]
    return msgspec.convert(data, type=RankQueryConfig)


def rank_query_config_to_dict(cfg: Any) -> dict[str, Any]:
    """Serialize RankQueryConfig to a plain dict (OpenAPI field names)."""
    return msgspec.to_builtins(cfg)


def convert_query_definition(obj: Any) -> QueryDefinition:
    if isinstance(obj, QueryDefinition):
        return obj
    if not isinstance(obj, dict):
        raise TypeError(f"QueryDefinition must be object, got {type(obj).__name__}")
    if "query" not in obj:
        raise TypeError("QueryDefinition requires 'query'")
    q = obj["query"]
    params = obj.get("parameters")
    param_defs: dict[str, ParameterDefinition] | None = None
    if params is not None:
        param_defs = {
            k: msgspec.convert(
                v if isinstance(v, dict) else {"default": v},
                type=ParameterDefinition,
            )
            for k, v in params.items()
        }
    if isinstance(q, str):
        return QueryDefinition(query=q, parameters=param_defs)
    return QueryDefinition(query=convert_rank_query_config(q), parameters=param_defs)
