"""Optional online text encoders (Hugging Face / sentence-transformers).

Library does **not** train models. Online encode is query-time only with an
LRU cache keyed by ``(model_name, text)``. Offline item vectors are written by
``examples/generator`` (or other offline jobs).

Call ``warm()`` (or ``get_encoder(..., warm=True)``) when building the
backend from engine YAML so the first query does not pay model-load latency.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from recql.catalog import EngineCatalog


from recql.encode.pooling import pool_vectors


def fake_embedding(text: str, dims: int = 8) -> list[float]:
    vec: list[float] = []
    for i in range(dims):
        h = hashlib.sha256(f"{text}:{i}".encode()).digest()
        val = (int.from_bytes(h[:4], "big") / 2**32) * 2 - 1
        vec.append(val)
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def vector_literal(vec: Sequence[float]) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in vec) + "]"


class TextEncoder(Protocol):
    model_name: str
    dims: int

    def encode(self, text: str) -> list[float]:
        ...

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def warm(self) -> None:
        """Eagerly load weights so first ``encode`` is cheap."""
        ...


@dataclass
class FakeTextEncoder:
    """Deterministic encoder for tests / CI (no HF download)."""

    model_name: str = "recql-fake"
    dims: int = 8

    def warm(self) -> None:
        return None

    def encode(self, text: str) -> list[float]:
        return fake_embedding(text, dims=self.dims)

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.encode(t) for t in texts]


@dataclass
class SentenceTransformerEncoder:
    """sentence-transformers wrapper (optional extra ``recql[encode]``)."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dims: int = 384
    _model: Any = field(default=None, repr=False)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for HF encode; "
                    "pip install 'recql[encode]'"
                ) from e
            self._model = SentenceTransformer(self.model_name)
            try:
                dim_fn = getattr(
                    self._model, "get_embedding_dimension", None
                ) or getattr(self._model, "get_sentence_embedding_dimension", None)
                if dim_fn is not None:
                    self.dims = int(dim_fn())
            except Exception:
                pass
        return self._model

    def warm(self) -> None:
        self._load()

    def encode(self, text: str) -> list[float]:
        model = self._load()
        vec = model.encode(str(text), normalize_embeddings=True)
        return [float(x) for x in list(vec)]

    def encode_many(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        if not texts:
            return []
        arr = model.encode(
            [str(t) for t in texts],
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in arr]


class LruEncodeCache:
    def __init__(self, maxsize: int = 2048) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(model_name: str, text: str) -> str:
        return hashlib.sha256(f"{model_name}\0{text}".encode()).hexdigest()

    def get_or_set(
        self, model_name: str, text: str, factory: Callable[[], list[float]]
    ) -> list[float]:
        key = self._key(model_name, text)
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                return list(self._data[key])
        value = factory()
        with self._lock:
            self._data[key] = list(value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
        return list(value)


_GLOBAL_CACHE = LruEncodeCache()
_ENCODER_POOL: dict[tuple[str, str, int], TextEncoder] = {}
_POOL_LOCK = threading.Lock()


def catalog_query_encoder_specs(
    catalog: EngineCatalog | None,
) -> list[dict[str, Any]]:
    """Embedding entries that imply an online query-text encoder."""
    if catalog is None:
        return []
    out: list[dict[str, Any]] = []
    for emb in catalog.embeddings.values():
        enc = emb.raw.get("encoder") if emb.raw else None
        if not isinstance(enc, dict):
            continue
        etype = str(enc.get("type") or "").lower()
        if etype in ("hugging_face", "huggingface", "sentence_transformers", "hf"):
            out.append(
                {
                    "name": emb.name,
                    "model_name": enc.get("model_name")
                    or enc.get("model_ref")
                    or emb.model_ref
                    or "sentence-transformers/all-MiniLM-L6-v2",
                    "dims": int(emb.dims or 384),
                    "encoder_type": etype,
                }
            )
    return out


def get_encoder(
    *,
    backend: str = "fake",
    model_name: str | None = None,
    dims: int = 8,
    warm: bool = False,
    reuse: bool = True,
) -> TextEncoder:
    """Select encoder: ``fake`` | ``sentence_transformers`` | ``auto``.

    Default is ``fake`` so CI/tests never download weights. Pass
    ``backend='sentence_transformers'`` or ``'auto'`` when HF is desired.
    Set ``warm=True`` to load weights immediately (registry / process start).
    """
    name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
    key = (backend, name, int(dims))

    def _build() -> TextEncoder:
        if backend == "fake":
            return FakeTextEncoder(model_name="recql-fake", dims=dims)
        if backend == "sentence_transformers":
            return SentenceTransformerEncoder(model_name=name, dims=dims)
        # auto: prefer ST if importable
        try:
            import sentence_transformers  # noqa: F401

            return SentenceTransformerEncoder(model_name=name, dims=dims)
        except ImportError:
            return FakeTextEncoder(model_name="recql-fake", dims=dims)

    if reuse:
        with _POOL_LOCK:
            enc = _ENCODER_POOL.get(key)
            if enc is None:
                enc = _build()
                _ENCODER_POOL[key] = enc
    else:
        enc = _build()

    if warm:
        enc.warm()
    return enc


def warm_encoders_for_catalog(
    catalog: EngineCatalog | None,
    *,
    backend: str = "fake",
    dims: int = 8,
) -> list[TextEncoder]:
    """Load encoders implied by engine YAML ``index.embeddings`` (once).

    When ``backend='fake'`` (CI default), still returns warmed fake encoders —
    never downloads HF. Real HF weights load only for ``sentence_transformers``
    / ``auto`` when the package is installed.
    """
    specs = catalog_query_encoder_specs(catalog)
    if not specs:
        enc = get_encoder(backend=backend, dims=dims, warm=True)
        return [enc]
    warmed: list[TextEncoder] = []
    seen: set[tuple[str, str, int]] = set()
    for spec in specs:
        model = str(spec["model_name"])
        d = int(spec["dims"] or dims)
        key = (backend, model, d)
        if key in seen:
            continue
        seen.add(key)
        warmed.append(
            get_encoder(backend=backend, model_name=model, dims=d, warm=True)
        )
    return warmed


def encode_query(
    text: str,
    *,
    encoder: TextEncoder | None = None,
    cache: LruEncodeCache | None = None,
) -> list[float]:
    enc = encoder or get_encoder(backend="fake")
    bag = cache or _GLOBAL_CACHE
    return bag.get_or_set(enc.model_name, str(text), lambda: enc.encode(str(text)))


__all__ = [
    "FakeTextEncoder",
    "LruEncodeCache",
    "SentenceTransformerEncoder",
    "TextEncoder",
    "catalog_query_encoder_specs",
    "encode_query",
    "fake_embedding",
    "get_encoder",
    "pool_vectors",
    "vector_literal",
    "warm_encoders_for_catalog",
]
