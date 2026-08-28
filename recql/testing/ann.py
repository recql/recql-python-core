"""Deterministic ANN helpers (hash embeddings + soft rank agreement).

Uses the same ``fake_embedding`` as CI seeds so expected rankings are stable.
Different engines' ANN indexes may shuffle near-ties; we assert soft overlap
against brute-force cosine over a fixed corpus — not exact id lists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

from recql.encode import fake_embedding

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_DEFAULT_CORPUS_PATH = _FIXTURES / "ann_corpus.json"


def load_ann_corpus(path: Path | None = None) -> list[dict[str, str]]:
    """Load ``[{id, title}, ...]`` used for deterministic vector checks."""
    raw = json.loads((path or _DEFAULT_CORPUS_PATH).read_text(encoding="utf-8"))
    return [{"id": str(row["id"]), "title": str(row["title"])} for row in raw]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)


def brute_force_vector_ids(
    query: str,
    corpus: Sequence[dict[str, str]],
    *,
    dims: int = 8,
    k: int = 10,
    text_key: str = "title",
) -> list[str]:
    """Exact ranking by cosine(fake_embedding(query), fake_embedding(item text))."""
    q = fake_embedding(query, dims=dims)
    scored: list[tuple[float, str]] = []
    for row in corpus:
        vec = fake_embedding(str(row[text_key]), dims=dims)
        scored.append((cosine(q, vec), str(row["id"])))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [i for _, i in scored[:k]]


def overlap_at_k(actual: Sequence[str], expected: Sequence[str], *, k: int) -> float:
    """``|actual[:k] ∩ expected[:k]| / k`` — soft ANN agreement score in ``[0, 1]``."""
    if k <= 0:
        return 1.0
    a = set(str(x) for x in actual[:k])
    e = set(str(x) for x in expected[:k])
    return len(a & e) / float(k)


def assert_ann_agreement(
    actual: Sequence[str],
    expected: Sequence[str],
    *,
    k: int = 10,
    min_overlap: float = 0.5,
    label: str = "ANN",
) -> float:
    """Require overlap@k ≥ ``min_overlap`` (default 50% of top-k).

    Returns the measured overlap for logging / secondary asserts.
    """
    score = overlap_at_k(actual, expected, k=k)
    if score + 1e-9 < min_overlap:
        raise AssertionError(
            f"{label} overlap@{k}={score:.3f} < min_overlap={min_overlap}; "
            f"actual={list(actual[:k])} expected={list(expected[:k])}"
        )
    return score
