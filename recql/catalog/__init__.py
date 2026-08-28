"""Engine YAML → EngineCatalog."""

from __future__ import annotations

import os
import re
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

_ENV_VAR_RE = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")


def expand_env_vars(val: Any) -> Any:
    """Expand ${VAR} or ${VAR:-default} in strings and nested structures."""
    if isinstance(val, str):
        def _repl(m: re.Match) -> str:
            var_name = m.group(1)
            default_val = m.group(2) if m.group(2) is not None else ""
            return os.environ.get(var_name, default_val)
        return _ENV_VAR_RE.sub(_repl, val)
    if isinstance(val, dict):
        return {k: expand_env_vars(v) for k, v in val.items()}
    if isinstance(val, list):
        return [expand_env_vars(v) for v in val]
    return val


@dataclass
class BackendSpec:
    name: str
    backend: str
    dsn: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingSpec:
    name: str
    dims: int | None = None
    encoder_type: str | None = None
    model_ref: str | None = None
    backend: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    name: str
    policy_type: str | None = None
    backend: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class LexicalSearchSpec:
    item_fields: list[str] = field(default_factory=list)
    user_fields: list[str] = field(default_factory=list)
    backend: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class EngineCatalog:
    name: str
    raw: dict[str, Any]
    backends: dict[str, BackendSpec] = field(default_factory=dict)
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

    def backend_for_embedding(self, name: str) -> str | None:
        emb = self.embeddings.get(name)
        if emb and emb.backend:
            return emb.backend
        stores = (self.raw.get("index") or {}).get("embedding_stores") or {}
        st = stores.get(name)
        if isinstance(st, dict) and st.get("backend"):
            return str(st["backend"])
        if len(self.backends) == 1:
            return next(iter(self.backends.keys()))
        return None

    def backend_for_lexical(self) -> str | None:
        if self.lexical_search and self.lexical_search.backend:
            return self.lexical_search.backend
        if len(self.backends) == 1:
            return next(iter(self.backends.keys()))
        return None

    def backend_for_model(self, name: str) -> str | None:
        m = self.models.get(name)
        if m and m.backend:
            return m.backend
        if len(self.backends) == 1:
            return next(iter(self.backends.keys()))
        return None

    def backend_for_entity(self, entity_role: str = "item") -> str | None:
        data_block = self.data.get(entity_role) or self.data.get(f"{entity_role}s")
        if isinstance(data_block, dict) and data_block.get("backend"):
            return str(data_block["backend"])
        if len(self.backends) == 1:
            return next(iter(self.backends.keys()))
        return None

    def is_multi_backend(self) -> bool:
        return len(self.backends) > 1

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

    raw = expand_env_vars(raw)

    name = str(raw.get("name") or raw.get("engine") or "default")
    cat = EngineCatalog(
        name=name,
        raw=raw,
        data=raw.get("data") or {},
        deployment=raw.get("deployment") or {},
        plugins=raw.get("plugins") or {},
    )

    # Parse backends block if present
    backends_dict = raw.get("backends")
    if not isinstance(backends_dict, dict):
        dep = raw.get("deployment") or {}
        backends_dict = dep.get("backends")
    if not isinstance(backends_dict, dict):
        plugins = raw.get("plugins") or {}
        backends_dict = plugins.get("backends")

    if isinstance(backends_dict, dict):
        for bname, bcfg in backends_dict.items():
            if isinstance(bcfg, dict):
                cat.backends[str(bname)] = BackendSpec(
                    name=str(bname),
                    backend=str(bcfg.get("backend") or bname),
                    dsn=str(bcfg.get("dsn")) if bcfg.get("dsn") is not None else None,
                    options={k: v for k, v in bcfg.items() if k not in ("backend", "dsn")},
                    raw=bcfg,
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
                backend=(str(emb["backend"]) if emb.get("backend") else None),
                raw=emb,
            )

    lex = index.get("lexical_search")
    if isinstance(lex, dict):
        cat.lexical_search = LexicalSearchSpec(
            item_fields=list(lex.get("item_fields") or []),
            user_fields=list(lex.get("user_fields") or []),
            backend=(str(lex["backend"]) if lex.get("backend") else None),
            raw=lex,
        )

    training = raw.get("training") or {}
    for m in training.get("models") or []:
        if isinstance(m, dict) and "name" in m:
            cat.models[m["name"]] = ModelSpec(
                name=m["name"],
                policy_type=m.get("type") or m.get("policy_type"),
                backend=(str(m["backend"]) if m.get("backend") else None),
                raw=m,
            )

    cat.queries = dict(raw.get("queries") or {})
    return cat


__all__ = [
    "BackendSpec",
    "DataBindings",
    "EmbeddingSpec",
    "EmbeddingStoreBinding",
    "EmbeddingStoreGroup",
    "EngineCatalog",
    "LexicalSearchSpec",
    "ModelSpec",
    "bindings_from_catalog",
    "default_fixture_bindings",
    "expand_env_vars",
    "load_engine_catalog",
]