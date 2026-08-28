"""Deterministic vector / ANN conformance (fake hash embeddings).

Exact top-1 for an identical title string is required (hash encoder is stable).
Top-k vs brute-force cosine may differ slightly across ANN implementations;
we require soft overlap@k (default ≥ 50%) rather than identical orderings.
"""

from __future__ import annotations

import pytest

from recql.testing.ann import assert_ann_agreement, brute_force_vector_ids
from recql.testing.features import VECTOR, require
from recql.testing.testbed import RecqlTestbed


@pytest.mark.asyncio
async def test_vector_exact_title_is_deterministic(recql_testbed: RecqlTestbed):
    """Query text == stored title text → fake embedding identity → top hit id 1."""
    require(recql_testbed, VECTOR)
    sql = """
    SELECT * FROM retrieve(
      text_search(
        query='Toy Story (1995)',
        mode='vector',
        text_embedding_ref='title_embedding',
        name='vec',
        limit=20
      )
    ) LIMIT 5
    """
    page = await recql_testbed.execute(sql)
    assert page.ids(), "vector retrieve returned no hits"
    assert page.ids()[0] == "1"


@pytest.mark.asyncio
async def test_vector_ann_agrees_with_bruteforce_fake_embeddings(
    recql_testbed: RecqlTestbed,
):
    """ANN top-k overlaps brute-force cosine(fake_embedding) within threshold.

    Near-ties and index approximations may reorder neighbors; overlap@k catches
    gross divergence while allowing small per-engine differences.
    """
    require(recql_testbed, VECTOR)
    query = "space adventure sci-fi"
    k = recql_testbed.ann_overlap_k
    corpus = recql_testbed.ann_corpus()
    expected = brute_force_vector_ids(query, corpus, dims=recql_testbed.dims, k=k)

    sql = f"""
    SELECT * FROM retrieve(
      text_search(
        query='{query}',
        mode='vector',
        text_embedding_ref='title_embedding',
        name='vec',
        limit={max(k * 2, 20)}
      )
    ) LIMIT {k}
    """
    page = await recql_testbed.execute(sql)
    actual = page.ids()
    assert actual, "vector retrieve returned no hits"

    # Restrict expected to ids the backend actually returned in a wider sense:
    # brute-force corpus is the conformance mini-set; seeded DBs may contain a
    # superset — compare only among corpus ids present in the hit list when
    # the engine returns extras from MovieLens.
    corpus_ids = {row["id"] for row in corpus}
    actual_in_corpus = [i for i in actual if i in corpus_ids]
    if len(actual_in_corpus) < max(3, k // 2):
        # Backend returned mostly non-corpus ids — still require top-1 of
        # brute force to appear somewhere in the page when the query is weak.
        assert expected[0] in actual or overlap_relaxed(actual, expected, k)
        return

    assert_ann_agreement(
        actual_in_corpus,
        expected,
        k=min(k, len(actual_in_corpus), len(expected)),
        min_overlap=recql_testbed.ann_min_overlap,
        label=f"{recql_testbed.backend} vector ANN",
    )


def overlap_relaxed(actual, expected, k: int) -> bool:
    from recql.testing.ann import overlap_at_k

    return overlap_at_k(actual, expected, k=k) >= 0.3
