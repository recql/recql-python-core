"""Comprehensive tests for interaction_pooling and split-store / cross-database similarity."""

import pytest
from recql.catalog import load_engine_catalog
from recql.execute.merge import Candidate, RetrieveBag
from recql.harness import recql
from recql.plugins.base import PluginRegistry, RetrieveRequest, Retriever
from recql.plugins.composite import CompositePluginRegistry


class MockVectorRetriever(Retriever):
    def __init__(
        self,
        name: str,
        *,
        vectors: dict[str, list[float]] | None = None,
        interactions: dict[str, list[str]] | None = None,
        search_candidates: list[str] | None = None,
    ) -> None:
        self.name = name
        self.vectors = vectors or {}
        self.interactions = interactions or {}
        self.search_candidates = search_candidates or ["item_result_1", "item_result_2"]
        self.received_query_vectors: list[list[float]] = []
        self.retrieved_calls: list[str] = []

    async def lookup_vector(
        self,
        embedding_ref: str,
        entity_type: str,
        entity_id: str,
        *,
        req: RetrieveRequest | None = None,
    ) -> list[float] | None:
        return self.vectors.get(entity_id)

    async def lookup_vectors(
        self,
        embedding_ref: str,
        entity_type: str,
        entity_ids: list[str],
        *,
        req: RetrieveRequest | None = None,
    ) -> dict[str, list[float]]:
        return {eid: self.vectors[eid] for eid in entity_ids if eid in self.vectors}

    async def lookup_interactions(
        self,
        user_id: str,
        limit: int = 10,
        *,
        req: RetrieveRequest | None = None,
    ) -> list[str]:
        items = self.interactions.get(user_id, [])
        return items[:limit]

    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        step = req.step
        self.retrieved_calls.append(self.name)
        qvec = getattr(step, "query_vector", None) or (
            req.params.get("__query_vector__") if req.params else None
        )
        if qvec:
            self.received_query_vectors.append(qvec)
        cands = [
            Candidate(
                id=cid,
                retrieval_score=0.95,
                attributes={"backend": self.name},
            )
            for cid in self.search_candidates
        ]
        return RetrieveBag(
            name=str(getattr(step, "name", "sim")),
            candidates=cands,
        )


@pytest.mark.asyncio
async def test_cross_backend_interaction_pooling():
    """Interaction history on Postgres (pg_db), item vectors on MariaDB (maria_db),

    and similarity search executes on Oracle (oracle_db).
    """
    yaml_config = """
name: split_interaction_pooling
backends:
  pg_db:
    backend: postgres
    dsn: postgresql://localhost/recql
  oracle_db:
    backend: oracle
    dsn: oracle://localhost/recql

data:
  interactions:
    backend: pg_db

index:
  embedding_stores:
    als:
      backend: oracle_db
"""
    cat = load_engine_catalog(yaml_config)

    # Setup 2 mock registries:
    # 1. Postgres has interaction history for user 'u1' -> items ['i1', 'i2', 'i3']
    pg_ret = MockVectorRetriever(
        "pg_db",
        interactions={"u1": ["i1", "i2", "i3"]},
    )

    # 2. Oracle has item vectors for i1, i2, i3 and runs ANN candidate retrieval
    oracle_ret = MockVectorRetriever(
        "oracle_db",
        vectors={
            "i1": [1.0, 0.0, 0.0],
            "i2": [0.0, 1.0, 0.0],
            "i3": [0.0, 0.0, 1.0],
        },
        search_candidates=["movie_101", "movie_102"],
    )

    composite_reg = CompositePluginRegistry(
        registries={
            "pg_db": PluginRegistry(retrievers={"similarity": pg_ret}),
            "oracle_db": PluginRegistry(retrievers={"similarity": oracle_ret}),
        },
        default_backend="oracle_db",
    )

    # Query with interaction_pooling encoder
    query = """
    SELECT *
    FROM similarity(embedding='als', encoder=interaction_pooling(input_user_id='u1', pooling_function='mean', truncate_interactions=10), limit=5)
    LIMIT 5
    """

    page = await recql(engine=cat, query=query, backend=composite_reg)

    # Check results
    assert page.ids() == ["movie_101", "movie_102"]

    # Verify search executed on oracle_db
    assert oracle_ret.retrieved_calls == ["oracle_db"]

    # Verify that oracle_ret received the mean pooled query vector ([1/3, 1/3, 1/3])
    assert len(oracle_ret.received_query_vectors) == 1
    qvec = oracle_ret.received_query_vectors[0]
    assert qvec == pytest.approx([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])


@pytest.mark.asyncio
async def test_cross_backend_precomputed_user_embedding():
    """User vectors stored on Postgres (pg_db), item candidate search on Oracle (oracle_db)."""
    yaml_config = """
name: split_user_embedding
backends:
  pg_db:
    backend: postgres
    dsn: postgresql://localhost/recql
  oracle_db:
    backend: oracle
    dsn: oracle://localhost/recql

index:
  embedding_stores:
    als:
      user:
        backend: pg_db
      item:
        backend: oracle_db
"""
    cat = load_engine_catalog(yaml_config)

    # 1. Postgres has user embedding for 'u42'
    pg_ret = MockVectorRetriever(
        "pg_db",
        vectors={"u42": [0.5, 0.5, 0.5]},
    )

    # 2. Oracle has item embeddings & performs ANN search
    oracle_ret = MockVectorRetriever(
        "oracle_db",
        search_candidates=["movie_500", "movie_501"],
    )

    composite_reg = CompositePluginRegistry(
        registries={
            "pg_db": PluginRegistry(retrievers={"similarity": pg_ret}),
            "oracle_db": PluginRegistry(retrievers={"similarity": oracle_ret}),
        },
        default_backend="oracle_db",
    )

    query = """
    SELECT *
    FROM similarity(embedding='als', encoder=precomputed_user(input_user_id='u42'), limit=5)
    LIMIT 5
    """

    page = await recql(engine=cat, query=query, backend=composite_reg)

    assert page.ids() == ["movie_500", "movie_501"]
    assert oracle_ret.retrieved_calls == ["oracle_db"]
    assert len(oracle_ret.received_query_vectors) == 1
    assert oracle_ret.received_query_vectors[0] == pytest.approx([0.5, 0.5, 0.5])


@pytest.mark.asyncio
async def test_cross_backend_interaction_pooling_empty_interactions():
    """If user has no interactions, interaction_pooling returns empty candidates."""
    yaml_config = """
name: split_empty_interactions
backends:
  pg_db:
    backend: postgres
    dsn: postgresql://localhost/recql
  oracle_db:
    backend: oracle
    dsn: oracle://localhost/recql

data:
  interactions:
    backend: pg_db

index:
  embedding_stores:
    als:
      backend: oracle_db
"""
    cat = load_engine_catalog(yaml_config)

    # User 'u_new' has no interactions
    pg_ret = MockVectorRetriever("pg_db", interactions={})
    oracle_ret = MockVectorRetriever("oracle_db", search_candidates=["should_not_reach"])

    composite_reg = CompositePluginRegistry(
        registries={
            "pg_db": PluginRegistry(retrievers={"similarity": pg_ret}),
            "oracle_db": PluginRegistry(retrievers={"similarity": oracle_ret}),
        },
        default_backend="oracle_db",
    )

    query = """
    SELECT *
    FROM similarity(embedding='als', encoder=interaction_pooling(input_user_id='u_new'), limit=5)
    LIMIT 5
    """

    page = await recql(engine=cat, query=query, backend=composite_reg)
    assert page.ids() == []
    # Oracle search was not even called
    assert len(oracle_ret.retrieved_calls) == 0


@pytest.mark.asyncio
async def test_cross_backend_interaction_pooling_max():
    """Test max pooling strategy across backends."""
    yaml_config = """
name: split_max_pooling
backends:
  pg_db:
    backend: postgres
  oracle_db:
    backend: oracle

data:
  interactions:
    backend: pg_db

index:
  embedding_stores:
    als:
      backend: oracle_db
"""
    cat = load_engine_catalog(yaml_config)

    pg_ret = MockVectorRetriever(
        "pg_db",
        interactions={"u2": ["i1", "i2"]},
    )
    oracle_ret = MockVectorRetriever(
        "oracle_db",
        vectors={
            "i1": [1.0, 5.0, -1.0],
            "i2": [4.0, 2.0, 3.0],
        },
        search_candidates=["cand_1"],
    )

    composite_reg = CompositePluginRegistry(
        registries={
            "pg_db": PluginRegistry(retrievers={"similarity": pg_ret}),
            "oracle_db": PluginRegistry(retrievers={"similarity": oracle_ret}),
        },
        default_backend="oracle_db",
    )

    query = """
    SELECT *
    FROM similarity(embedding='als', encoder=interaction_pooling(input_user_id='u2', pooling_function='max', truncate_interactions=5), limit=5)
    LIMIT 5
    """

    page = await recql(engine=cat, query=query, backend=composite_reg)
    assert page.ids() == ["cand_1"]
    assert len(oracle_ret.received_query_vectors) == 1
    # Max of [1.0, 5.0, -1.0] and [4.0, 2.0, 3.0] is [4.0, 5.0, 3.0]
    assert oracle_ret.received_query_vectors[0] == pytest.approx([4.0, 5.0, 3.0])
