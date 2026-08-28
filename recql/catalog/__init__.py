"""Engine YAML → EngineCatalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from recql.catalog.bindings import (
    DataBindings,
    EmbeddingStoreBinding,
    EmbeddingStoreGroup,
    PersonalFilterBinding,
    bindings_from_catalog,
    default_fixture_bindings,
)


@dataclass
class EmbeddingSpec:
    name: str
    dims: int | None = None
    encoder_type: str | None = None
    model_ref: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    name: str
    policy_type: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LexicalSearchSpec:
    item_fields: list[str] = field(default_factory=list)
    user_fields: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineCatalog:
    name: str
    raw: dict[str, Any]
    embeddings: dict[str, EmbeddingSpec] = field(default_factory=dict)
    models: dict[str, ModelSpec] = field(default_factory=dict)
    lexical_search: LexicalSearchSpec | None = None
    queries: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    deployment: dict[str, Any] = field(default_factory=dict)
    plugins: dict[str, Any] = field(default_factory=dict)

    def embedding(self, name: str) -> EmbeddingSpec | None:
        return self.embeddings.get(name)

    def model(self, name: str) -> ModelSpec | None:
        return self.models.get(name)

    def bindings(self) -> DataBindings:
        return bindings_from_catalog(self)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_yaml_file(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        return {}
    base = Path(path).parent
    include = loaded.pop("include", None) or loaded.pop("includes", None)
    if not include:
        return loaded
    files = [include] if isinstance(include, str) else list(include)
    merged: dict[str, Any] = {}
    for rel in files:
        part = _load_yaml_file((base / rel).resolve())
        merged = _deep_merge(merged, part)
    return _deep_merge(merged, loaded)


def load_engine_catalog(source: str | Path | dict[str, Any]) -> EngineCatalog:
    if isinstance(source, dict):
        raw = source
    elif isinstance(source, Path) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        raw = _load_yaml_file(Path(source))
    else:
        raw = yaml.safe_load(str(source))

    name = str(raw.get("name") or raw.get("engine") or "default")
    cat = EngineCatalog(
        name=name,
        raw=raw,
        data=raw.get("data") or {},
        deployment=raw.get("deployment") or {},
        plugins=raw.get("plugins") or {},
    )

    index = raw.get("index") or {}
    for emb in index.get("embeddings") or []:
        if isinstance(emb, dict) and "name" in emb:
            enc = emb.get("encoder") or {}
            cat.embeddings[emb["name"]] = EmbeddingSpec(
                name=emb["name"],
                dims=emb.get("dims") or emb.get("embedding_dim"),
                encoder_type=(enc.get("type") if isinstance(enc, dict) else None),
                model_ref=(enc.get("model_ref") if isinstance(enc, dict) else None),
                raw=emb,
            )

    lex = index.get("lexical_search")
    if isinstance(lex, dict):
        cat.lexical_search = LexicalSearchSpec(
            item_fields=list(lex.get("item_fields") or []),
            user_fields=list(lex.get("user_fields") or []),
            raw=lex,
        )

    training = raw.get("training") or {}
    for m in training.get("models") or []:
        if isinstance(m, dict) and "name" in m:
            cat.models[m["name"]] = ModelSpec(
                name=m["name"],
                policy_type=m.get("type") or m.get("policy_type"),
                raw=m,
            )

    cat.queries = dict(raw.get("queries") or {})
    return cat


__all__ = [
    "DataBindings",
    "EmbeddingSpec",
    "EmbeddingStoreBinding",
    "EmbeddingStoreGroup",
    "EngineCatalog",
    "LexicalSearchSpec",
    "ModelSpec",
    "bindings_from_catalog",
    "default_fixture_bindings",
    "load_engine_catalog",
]