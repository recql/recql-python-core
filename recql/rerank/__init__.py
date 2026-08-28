"""Optional zero-shot text rerankers: colbert_v2 / cross_encoder.

Models are **loaded**, never trained, by the library. Training / fine-tune
belongs in offline jobs (``examples/generator``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from recql.pyutils.jaccard import jaccard


class TextReranker(Protocol):
    name: str

    def score(self, item_text: str, query: str) -> float:
        ...


@dataclass
class JaccardReranker:
    """Cheap fallback when HF rerank packages are not installed."""

    name: str = "jaccard"

    def score(self, item_text: str, query: str) -> float:
        return jaccard(set(item_text.lower()), set(query.lower()))


@dataclass
class CrossEncoderReranker:
    """sentence-transformers CrossEncoder (optional ``recql[encode]``)."""

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    name: str = "cross_encoder"
    _model: Any = field(default=None, repr=False)

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers required for cross_encoder; "
                    "pip install 'recql[encode]'"
                ) from e
            self._model = CrossEncoder(self.model_name)
        return self._model

    def score(self, item_text: str, query: str) -> float:
        model = self._load()
        pred = model.predict([(str(query), str(item_text))])
        val = pred[0] if hasattr(pred, "__len__") else pred
        return float(val)


@dataclass
class ColbertV2Reranker:
    """ColBERT-style late interaction via token embeddings when available.

    Preferred path: ``ragatouille`` / colbert models if installed. Otherwise
    falls back to mean-pooled sentence-transformer cosine (documented soft path).
    """

    model_name: str = "colbert-ir/colbertv2.0"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    name: str = "colbert_v2"
    _st: Any = field(default=None, repr=False)
    _backend: str = field(default="unset", repr=False)

    def _ensure(self) -> None:
        if self._backend != "unset":
            return
        try:
            from ragatouille import RAGPretrainedModel  # type: ignore

            self._st = RAGPretrainedModel.from_pretrained(self.model_name)
            self._backend = "ragatouille"
            return
        except Exception:
            pass
        try:
            from sentence_transformers import SentenceTransformer

            self._st = SentenceTransformer(self.embed_model)
            self._backend = "st_cosine"
        except ImportError:
            self._st = None
            self._backend = "jaccard"

    def score(self, item_text: str, query: str) -> float:
        self._ensure()
        if self._backend == "ragatouille" and self._st is not None:
            # RAGPretrainedModel exposes ranking helpers; use embed cosine as stable API.
            try:
                results = self._st.rerank(query=str(query), documents=[str(item_text)], k=1)
                if results:
                    return float(results[0].get("score") or results[0][1])
            except Exception:
                pass
        if self._backend == "st_cosine" and self._st is not None:
            import math

            qv = list(self._st.encode(str(query), normalize_embeddings=True))
            iv = list(self._st.encode(str(item_text), normalize_embeddings=True))
            return float(sum(a * b for a, b in zip(qv, iv, strict=True)))
        return JaccardReranker().score(item_text, query)


_RERANKERS: dict[str, TextReranker] = {}


def get_reranker(name: str, *, prefer_real: bool | None = None) -> TextReranker:
    """Return a reranker. Default is Jaccard unless ``RECQL_RERANK_REAL=1``.

    Real CrossEncoder / ColBERT downloads are opt-in to keep CI offline-safe.
    """
    import os

    key = name.lower()
    if key in _RERANKERS:
        return _RERANKERS[key]
    if prefer_real is None:
        prefer_real = os.environ.get("RECQL_RERANK_REAL", "").lower() in (
            "1",
            "true",
            "yes",
        )
    if not prefer_real:
        r: TextReranker = JaccardReranker(name=key)
        _RERANKERS[key] = r
        return r
    if key in ("cross_encoder", "cross-encoder"):
        try:
            import sentence_transformers  # noqa: F401

            r = CrossEncoderReranker()
            _RERANKERS[key] = r
            return r
        except ImportError:
            r = JaccardReranker(name="cross_encoder")
            _RERANKERS[key] = r
            return r
    if key in ("colbert_v2", "colbert", "colbertv2"):
        r = ColbertV2Reranker()
        _RERANKERS[key] = r
        return r
    r = JaccardReranker(name=key)
    _RERANKERS[key] = r
    return r


def score_rerank(name: str, item_text: str, query: str) -> float:
    return get_reranker(name).score(item_text, query)
