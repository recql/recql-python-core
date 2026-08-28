# recql-python-core

Storage-agnostic RecQL engine. No database drivers. Backend packs register via
entry points (`recql.backends` / `recql.dialects` / `recql.connectors`); core
does not hardcode backend names or aliases.

---

## Engine Configuration (`engine.yaml`)

RecQL engines are configured via declarative YAML files describing the database backends, data tables, search indexes, embeddings, and ranking models.

### Single-Backend Engine Configuration

For single-database setups, set `plugins.backend` to the target database dialect:

```yaml
version: v2
name: my_single_db_engine

plugins:
  backend: postgres
  postgres:
    encode_backend: sentence_transformers
    lexical_backend: auto

data:
  item_table:
    name: items
    id_column: item_id
    attrs_column: attrs
  user_table:
    name: users
    id_column: user_id
  interaction_table:
    name: interactions
    user_id_column: user_id
    item_id_column: item_id
  filters:
    - name: exclude_seen
      filter_type: personal_filter
      filter_table:
        type: query
        query: SELECT item_id, user_id FROM interactions

index:
  embeddings:
    - name: content_embedding
      dims: 384
      encoder:
        type: sentence_transformers
        model_ref: all-MiniLM-L6-v2
    - name: als
      dims: 32
      encoder:
        type: precomputed_user

  lexical_search:
    item_fields: [title, overview]

training:
  models:
    - name: ranker_lgbm
      type: lgbm
```

---

### Multi-Backend (Federated) Engine Configuration

RecQL natively supports multi-backend federation, allowing a single query to route across multiple databases concurrently (e.g. vector search on Oracle 23ai, collaborative filtering on PostgreSQL, and lexical search on MariaDB).

#### 1. `backends:` Declaration Block

Define named backend connection pools. Environment variables (`${VAR}` or `${VAR:-default}`) are expanded automatically:

```yaml
version: v2
name: federated_recommender_engine

backends:
  primary_pg:
    backend: postgres
    dsn: ${PG_DSN:-postgresql://recql:recql@127.0.0.1:5432/recql}
    min_size: 2
    max_size: 10

  vector_oracle:
    backend: oracle
    dsn: ${ORACLE_DSN:-recql/recql@127.0.0.1:1521/FREEPDB1}

  search_mariadb:
    backend: mariadb
    dsn: ${MARIADB_DSN:-recql:recql@127.0.0.1:3306/recql}
```

#### 2. Assigning Backends to Tables, Embeddings, Models, & Filters

Assign components to specific backends via the `backend:` key:

```yaml
data:
  # Relational entities live in primary PostgreSQL
  item_table:
    backend: primary_pg
    name: items
    id_column: item_id
    attrs_column: attrs

  user_table:
    backend: primary_pg
    name: users
    id_column: user_id

  interaction_table:
    backend: primary_pg
    name: interactions
    user_id_column: user_id
    item_id_column: item_id

  filters:
    - name: exclude_seen
      backend: primary_pg
      filter_table:
        type: query
        query: SELECT item_id, user_id FROM interactions

index:
  embeddings:
    # 384-d semantic vectors routed to Oracle 23ai AI Vector Search
    - name: content_embedding
      backend: vector_oracle
      dims: 384
      encoder:
        type: sentence_transformers
        model_ref: all-MiniLM-L6-v2

    # 32-d user/item collaborative filtering vectors routed to PostgreSQL
    - name: als
      backend: primary_pg
      dims: 32
      encoder:
        type: precomputed_user

  # Full-text BM25 / lexical search routed to MariaDB FullText
  lexical_search:
    backend: search_mariadb
    item_fields: [title, overview]

training:
  models:
    # Scoring model stored and evaluated via primary PostgreSQL
    - name: ranker_lgbm
      backend: primary_pg
```

---

### Python API: Multi-Backend Connection & Execution

Use `open_engine` to open connection pools across all configured backends and execute queries:

```python
from recql import load_engine_catalog, recql
from recql.plugins.connectors import open_engine

# 1. Load the multi-backend catalog
catalog = load_engine_catalog("engine.yaml")

# 2. Open all connection pools (returns CompositePluginRegistry + close handler)
registry, close_engine = await open_engine(catalog)

try:
    # 3. Execute RecQL query — retrievers fan out across backends automatically
    query = """
    SELECT *
    FROM retrieve(
      text_search(query=$query_text, mode=vector(text_embedding_ref='content_embedding'), limit=50),
      similarity(embedding_ref='als', encoder=precomputed_user(input_user_id=$user_id), limit=50),
      text_search(query=$query_text, mode='lexical', limit=50)
    )
    WHERE prebuilt(filter_ref='exclude_seen', input_user_id=$user_id)
    ORDER BY score(model='ranker_lgbm') DESC
    LIMIT 20;
    """

    page = await recql(
        engine=catalog,
        query=query,
        params={"query_text": "sci-fi adventure", "user_id": "42"},
        backend=registry,
    )

    for item in page.items:
        print(item.id, item.attributes)
finally:
    await close_engine()
```

---

## Conformance suite

Backend packs must pass the same suite under `recql.testing.conformance`.

1. Implement a pytest fixture named **`recql_testbed`** yielding `recql.testing.RecqlTestbed`.
2. Star-import the suite into the pack's `tests/test_conformance.py`.
3. Run against a live DB started from that pack’s `docker-compose.yml`.

```python
from recql.testing import RecqlTestbed, SQL_BACKEND_FEATURES

@pytest.fixture
async def recql_testbed():
    # open pool, seed via recql-playground, build registry
    yield RecqlTestbed(
        backend=plugin_backend_name(catalog),  # from engine YAML
        registry=registry,
        catalog=catalog,
        popular_rank_column="_derived_popular_rank",
        features=SQL_BACKEND_FEATURES,
    )
```

Core’s own `tests/` are **unit-only** (no live DB). Do not run conformance from this repo without a backend fixture.

## Install

```bash
pip install "recql @ git+https://github.com/recql/recql-python-core.git"
pip install -e ".[dev]"   # unit tests
pytest -q
```
