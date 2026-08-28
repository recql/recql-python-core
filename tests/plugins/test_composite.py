"""Tests for CompositePluginRegistry and multi-backend routing."""

import pytest
from recql.catalog import load_engine_catalog
from recql.execute.merge import Candidate, RetrieveBag
from recql.harness import recql
from recql.plugins.base import PluginRegistry, RetrieveRequest, Retriever, Scorer
from recql.plugins.composite import CompositePluginRegistry


class MockNamedRetriever(Retriever):
    def __init__(self, backend_tag: str, canned_results: dict[str, list[str]]) -> None:
        self.backend_tag = backend_tag
        self.canned_results = canned_results
        self.calls: list[str] = []

    async def retrieve(self, req: RetrieveRequest) -> RetrieveBag:
        step = req.step
        emb = getattr(step, "embedding_ref", None)
        mode = getattr(step, "mode", None)
        mode_str = str(getattr(mode, "type", mode) or "")
        key = str(emb or mode_str or getattr(step, "type", "retrieve"))
        self.calls.append(key)
        ids = self.canned_results.get(key, ["default_id"])
        cands = [
            Candidate(id=cid, retrieval_score=0.9, attributes={"source_backend": self.backend_tag})
            for cid in ids
        ]
        return RetrieveBag(name=str(getattr(step, "name", "bag")), candidates=cands)


class MockScorer(Scorer):
    def __init__(self, backend_tag: str) -> None:
        self.backend_tag = backend_tag
        self.scored_models: list[str] = []

    async def score_many(self, plan, candidates, ctx):
        mname = getattr(plan, "value_model", None) or getattr(plan, "model", "default_model")
        self.scored_models.append(str(mname))
        return [1.0] * len(candidates)


@pytest.mark.asyncio
async def test_composite_retriever_routing():
    yaml_doc = """
name: multi_db_recs
backends:
  postgres_store:
    backend: postgres
    dsn: postgresql://localhost/recql
  oracle_vector_store:
    backend: oracle
    dsn: oracle://localhost/recql
  mariadb_search_store:
    backend: mariadb
    dsn: mysql://localhost/recql

index:
  embeddings:
    - name: content_embedding
      backend: postgres_store
      dims: 384
    - name: als
      backend: oracle_vector_store
      dims: 32

  lexical_search:
    backend: mariadb_search_store
    item_fields: [title, overview]

training:
  models:
    - name: click_model
      backend: postgres_store
"""
    cat = load_engine_catalog(yaml_doc)

    pg_ret = MockNamedRetriever("postgres", {"content_embedding": ["pg_1", "pg_2"]})
    ora_ret = MockNamedRetriever("oracle", {"als": ["ora_1", "ora_2"]})
    maria_ret = MockNamedRetriever("mariadb", {"lexical": ["maria_1", "maria_2"], "text_search": ["maria_1", "maria_2"]})

    pg_reg = PluginRegistry(
        retrievers={"similarity": pg_ret, "text_search": pg_ret},
        scorers={"score": MockScorer("postgres")},
    )
    ora_reg = PluginRegistry(
        retrievers={"similarity": ora_ret, "text_search": ora_ret},
    )
    maria_reg = PluginRegistry(
        retrievers={"similarity": maria_ret, "text_search": maria_ret},
    )

    composite_reg = CompositePluginRegistry(
        registries={
            "postgres_store": pg_reg,
            "oracle_vector_store": ora_reg,
            "mariadb_search_store": maria_reg,
        },
        default_backend="postgres_store",
    )

    query = """
    SELECT *
    FROM retrieve(
        similarity(embedding_ref='content_embedding', limit=10),
        similarity(embedding_ref='als', limit=10),
        text_search(query='comedy', mode='lexical', limit=10)
    )
    LIMIT 20;
    """

    page = await recql(
        engine=cat,
        query=query,
        backend=composite_reg,
    )

    assert "content_embedding" in pg_ret.calls
    assert "als" in ora_ret.calls
    assert len(maria_ret.calls) > 0

    returned_ids = page.ids()
    assert "pg_1" in returned_ids
    assert "ora_1" in returned_ids
    assert "maria_1" in returned_ids


@pytest.mark.asyncio
async def test_composite_registry_dict_in_harness():
    yaml_doc = """
name: dual_db
backends:
  pg:
    backend: postgres
    dsn: postgresql://localhost/recql
  ora:
    backend: oracle
    dsn: oracle://localhost/recql

index:
  embeddings:
    - name: text_emb
      backend: pg
      dims: 384
    - name: cf_emb
      backend: ora
      dims: 32
"""
    cat = load_engine_catalog(yaml_doc)

    pg_ret = MockNamedRetriever("pg", {"text_emb": ["item_pg"]})
    ora_ret = MockNamedRetriever("ora", {"cf_emb": ["item_ora"]})

    pg_reg = PluginRegistry(retrievers={"similarity": pg_ret})
    ora_reg = PluginRegistry(retrievers={"similarity": ora_ret})

    # Pass backend as a dict mapping backend names to registries
    page = await recql(
        engine=cat,
        query="""
        SELECT *
        FROM retrieve(
            similarity(embedding_ref='text_emb', limit=5),
            similarity(embedding_ref='cf_emb', limit=5)
        )
        LIMIT 10;
        """,
        backend={"pg": pg_reg, "ora": ora_reg},
    )

    assert "item_pg" in page.ids()
    assert "item_ora" in page.ids()


@pytest.mark.asyncio
async def test_composite_text_search_vector_routing():
    yaml_doc = """
name: text_search_routing
backends:
  pg:
    backend: postgres
    dsn: postgresql://localhost/recql
  ora:
    backend: oracle
    dsn: oracle://localhost/recql

index:
  embeddings:
    - name: title_embedding
      backend: ora
      dims: 384
"""
    cat = load_engine_catalog(yaml_doc)

    pg_ret = MockNamedRetriever("pg", {"vector": ["pg_vec"]})
    ora_ret = MockNamedRetriever("ora", {"vector": ["ora_vec"]})

    pg_reg = PluginRegistry(retrievers={"text_search": pg_ret})
    ora_reg = PluginRegistry(retrievers={"text_search": ora_ret})

    composite_reg = CompositePluginRegistry(
        registries={"pg": pg_reg, "ora": ora_reg},
        default_backend="pg",
    )

    page = await recql(
        engine=cat,
        query="""
        SELECT *
        FROM retrieve(
            text_search(query='sci-fi', mode=vector(text_embedding_ref='title_embedding'), limit=5)
        )
        LIMIT 5;
        """,
        backend=composite_reg,
    )

    assert "ora_vec" in page.ids()
    assert "vector" in ora_ret.calls


@pytest.mark.asyncio
async def test_composite_scoring_and_filtering_routing():
    from recql.plugins.base import FilterPlugin

    class MockFilter(FilterPlugin):
        def __init__(self, tag: str) -> None:
            self.tag = tag
            self.applied = False

        async def apply(self, step, rows, ctx):
            self.applied = True
            for r in rows:
                r.attributes[f"filtered_by_{self.tag}"] = True
            return rows

    yaml_doc = """
name: score_and_filter
backends:
  primary_db:
    backend: postgres
    dsn: postgresql://localhost/recql

index:
  embeddings:
    - name: content
      backend: primary_db
      dims: 384

data:
  filters:
    - name: exclude_seen
      backend: primary_db

training:
  models:
    - name: lgbm_ranker
      backend: primary_db
"""
    cat = load_engine_catalog(yaml_doc)

    pg_ret = MockNamedRetriever("primary_db", {"content": ["item_1", "item_2"]})
    pg_scorer = MockScorer("primary_db")
    pg_filter = MockFilter("primary_db")

    pg_reg = PluginRegistry(
        retrievers={"similarity": pg_ret},
        scorers={"score": pg_scorer},
        filters={"exclude_seen": pg_filter},
    )

    composite_reg = CompositePluginRegistry(
        registries={"primary_db": pg_reg},
        default_backend="primary_db",
    )

    ir_query = {
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "content",
                "query_encoder": {
                    "type": "precomputed_item",
                    "input_item_id": "100",
                },
                "limit": 10,
            }
        ],
        "filter": [
            {
                "type": "prebuilt",
                "filter_ref": "exclude_seen",
            }
        ],
        "score": {
            "type": "score_ensemble",
            "value_model": "lgbm_ranker",
        },
        "limit": 5,
    }

    page = await recql(
        engine=cat,
        query=ir_query,
        backend=composite_reg,
    )

    assert len(page.items) == 2
    assert pg_filter.applied is True
    assert "lgbm_ranker" in pg_scorer.scored_models


@pytest.mark.asyncio
async def test_open_engine_multi_backend(monkeypatch):
    from recql.plugins.connectors import open_engine

    opened_connections: list[tuple[str, str]] = []
    closed_connections: list[str] = []

    async def mock_open_connection(dsn, *, backend, catalog=None, **kwargs):
        opened_connections.append((backend, dsn))
        reg = PluginRegistry()

        async def close():
            closed_connections.append(backend)

        return reg, close

    monkeypatch.setattr("recql.plugins.connectors.open_connection", mock_open_connection)

    yaml_doc = """
name: federated_test
backends:
  pg_main:
    backend: postgres
    dsn: postgresql://localhost/recql
  ora_main:
    backend: oracle
    dsn: oracle://localhost/recql
"""
    cat = load_engine_catalog(yaml_doc)
    composite_reg, close_all = await open_engine(cat)

    assert isinstance(composite_reg, CompositePluginRegistry)
    assert len(opened_connections) == 2
    assert ("postgres", "postgresql://localhost/recql") in opened_connections
    assert ("oracle", "oracle://localhost/recql") in opened_connections

    await close_all()
    assert len(closed_connections) == 2
    assert "postgres" in closed_connections
    assert "oracle" in closed_connections


@pytest.mark.asyncio
async def test_composite_explicit_step_backend_override():
    yaml_doc = """
name: override_test
backends:
  pg_db:
    backend: postgres
    dsn: postgresql://localhost/recql
  ora_db:
    backend: oracle
    dsn: oracle://localhost/recql

index:
  embeddings:
    - name: text_emb
      backend: pg_db
      dims: 384
"""
    cat = load_engine_catalog(yaml_doc)

    pg_ret = MockNamedRetriever("pg_db", {"text_emb": ["pg_val"]})
    ora_ret = MockNamedRetriever("ora_db", {"text_emb": ["ora_val"]})

    composite_reg = CompositePluginRegistry(
        registries={
            "pg_db": PluginRegistry(retrievers={"similarity": pg_ret}),
            "ora_db": PluginRegistry(retrievers={"similarity": ora_ret}),
        },
        default_backend="pg_db",
    )

    # IR query with explicit backend override on the step
    ir_query = {
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "text_emb",
                "backend": "ora_db",
                "query_encoder": {
                    "type": "precomputed_item",
                    "input_item_id": "1",
                },
                "limit": 5,
            }
        ],
        "limit": 5,
    }

    page = await recql(engine=cat, query=ir_query, backend=composite_reg)
    assert page.ids() == ["ora_val"]
    assert "text_emb" in ora_ret.calls
    assert len(pg_ret.calls) == 0


@pytest.mark.asyncio
async def test_composite_full_lifecycle_multi_backend():
    from recql.plugins.mock import InMemoryKvStore

    yaml_doc = """
name: lifecycle_test
backends:
  primary_store:
    backend: postgres
    dsn: postgresql://localhost/recql
  vector_store:
    backend: oracle
    dsn: oracle://localhost/recql
  scoring_store:
    backend: mariadb
    dsn: mysql://localhost/recql

index:
  embeddings:
    - name: user_cf
      backend: vector_store
      dims: 64

training:
  models:
    - name: ranker_v1
      backend: scoring_store
"""
    cat = load_engine_catalog(yaml_doc)

    # Vector store retrieves candidates
    vec_ret = MockNamedRetriever("vector_store", {"user_cf": ["item_1", "item_2", "item_3", "item_4"]})
    vec_reg = PluginRegistry(retrievers={"similarity": vec_ret})

    # Scoring store scores candidates
    score_scorer = MockScorer("scoring_store")
    score_reg = PluginRegistry(scorers={"score": score_scorer, "score_ensemble": score_scorer})

    # Primary store manages pagination KV
    kv = InMemoryKvStore()
    primary_reg = PluginRegistry(kv=kv)

    composite_reg = CompositePluginRegistry(
        registries={
            "primary_store": primary_reg,
            "vector_store": vec_reg,
            "scoring_store": score_reg,
        },
        default_backend="primary_store",
        kv=kv,
    )

    ir_query = {
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "user_cf",
                "query_encoder": {
                    "type": "precomputed_user",
                    "input_user_id": "u1",
                },
                "limit": 10,
            }
        ],
        "score": {
            "type": "score_ensemble",
            "value_model": "ranker_v1",
        },
        "limit": 2,
    }

    # First page
    page1 = await recql(
        engine=cat,
        query=ir_query,
        backend=composite_reg,
        pagination_key="user_u1_session",
    )
    assert page1.ids() == ["item_1", "item_2"]
    assert "ranker_v1" in score_scorer.scored_models

    # Second page (items 1 & 2 excluded via KV)
    page2 = await recql(
        engine=cat,
        query=ir_query,
        backend=composite_reg,
        pagination_key="user_u1_session",
    )
    assert page2.ids() == ["item_3", "item_4"]


@pytest.mark.asyncio
async def test_composite_parallel_retrieve_error_handling():
    from recql.errors import ExecuteError

    class FailingRetriever(Retriever):
        async def retrieve(self, req: RetrieveRequest):
            raise ExecuteError("Database connection lost during query execution")

    failing_reg = PluginRegistry(retrievers={"similarity": FailingRetriever()})
    good_ret = MockNamedRetriever("good_db", {"content": ["ok_1"]})
    good_reg = PluginRegistry(retrievers={"similarity": good_ret})

    composite_reg = CompositePluginRegistry(
        registries={"db_fail": failing_reg, "db_good": good_reg},
        default_backend="db_good",
    )

    yaml_doc = """
name: fail_test
backends:
  db_fail:
    backend: postgres
    dsn: postgresql://localhost/recql
  db_good:
    backend: oracle
    dsn: oracle://localhost/recql

index:
  embeddings:
    - name: bad_emb
      backend: db_fail
      dims: 16
    - name: good_emb
      backend: db_good
      dims: 16
"""
    cat = load_engine_catalog(yaml_doc)

    ir_query = {
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "bad_emb",
                "query_encoder": {"type": "precomputed_item", "input_item_id": "1"},
                "limit": 5,
            },
            {
                "type": "similarity",
                "embedding_ref": "good_emb",
                "query_encoder": {"type": "precomputed_item", "input_item_id": "1"},
                "limit": 5,
            },
        ],
        "limit": 10,
    }

    with pytest.raises(ExecuteError, match="Database connection lost"):
        await recql(engine=cat, query=ir_query, backend=composite_reg)



