"""Bind RankQueryConfig + EngineCatalog → BoundRankQuery."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recql.catalog import EngineCatalog
from recql.errors import BindError
from recql.openapi_ir import RankQueryConfig, rank_query_config_to_dict
from recql.openapi_ir.models import (
    CandidateAttributesRetrieveStep,
    CandidateIdsRetrieveStep,
    ColumnOrderRetrieveStep,
    FilterRetrieveStep,
    SimilarityRetrieveStep,
    TextSearchRetrieveStep,
    VectorSearchMode,
)


@dataclass
class BoundRankQuery:
    config: RankQueryConfig
    catalog: EngineCatalog | None
    params: dict[str, Any] = field(default_factory=dict)
    diagnostics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return rank_query_config_to_dict(self.config)


def bind(
    config: RankQueryConfig,
    catalog: EngineCatalog | None = None,
    *,
    params: dict[str, Any] | None = None,
    require_catalog: bool = False,
) -> BoundRankQuery:
    diagnostics: list[str] = []
    if require_catalog and catalog is None:
        raise BindError("engine catalog required")

    retrieve = list(config.retrieve or [])
    if not retrieve:
        diagnostics.append("warning: empty retrieve list")

    for step in retrieve:
        _bind_retrieve_step(step, catalog, diagnostics)

    # Artifact version / dims pins from deployment
    if catalog is not None:
        from recql.artifacts import pins_from_deployment

        pins = pins_from_deployment(catalog.deployment)
        if pins:
            diagnostics.append(f"artifact_pins={pins}")
        for emb_name, spec in catalog.embeddings.items():
            if spec.dims is not None:
                diagnostics.append(f"embedding {emb_name} dims={spec.dims}")

    score = config.score
    if score is not None and getattr(score, "type", None) == "score_ensemble":
        vm = getattr(score, "value_model", "")
        if catalog is not None and vm and vm in catalog.models:
            diagnostics.append(f"bound model {vm}")

    entity = getattr(config, "from_", None) or getattr(config, "from", None)
    # msgspec rename: access via to_builtins
    d = rank_query_config_to_dict(config)
    entity = d.get("from")
    if entity not in (None, "item", "user", "item_attribute"):
        raise BindError(f"invalid entity from={entity!r}")

    return BoundRankQuery(
        config=config,
        catalog=catalog,
        params=dict(params or {}),
        diagnostics=diagnostics,
    )


def _bind_retrieve_step(
    step: Any, catalog: EngineCatalog | None, diagnostics: list[str]
) -> None:
    t = getattr(step, "type", None)
    if t == "similarity":
        assert isinstance(step, SimilarityRetrieveStep) or hasattr(step, "embedding_ref")
        ref = step.embedding_ref
        if catalog is not None and catalog.embedding(ref) is None:
            raise BindError(f"unknown embedding_ref: {ref}")
    elif t == "text_search":
        mode = step.mode
        if isinstance(mode, VectorSearchMode) or getattr(mode, "type", None) == "vector":
            ref = getattr(mode, "text_embedding_ref", None)
            if catalog is not None and ref and catalog.embedding(ref) is None:
                raise BindError(f"unknown text_embedding_ref: {ref}")
        if getattr(mode, "type", None) == "lexical" or (
            isinstance(mode, dict) and mode.get("type") == "lexical"
        ):
            if catalog is not None and catalog.lexical_search is None:
                diagnostics.append("warning: lexical_search not configured in catalog")
    elif t in (
        "column_order",
        "filter",
        "candidate_ids",
        "candidate_attributes",
    ):
        pass
    else:
        raise BindError(f"unknown retrieve type: {t}")

    where = getattr(step, "where", None)
    if where is not None:
        # Fail closed is enforced at plugin retrieve time if unsupported;
        # bind records that a prefilter is present.
        diagnostics.append(f"prefilter present on {t} name={getattr(step, 'name', None)}")
