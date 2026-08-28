"""Tests for multi-backend engine catalog parsing and bindings."""

import os
from recql.catalog import load_engine_catalog, expand_env_vars


def test_expand_env_vars():
    os.environ["RECQL_TEST_VAR"] = "my_custom_value"
    assert expand_env_vars("${RECQL_TEST_VAR}") == "my_custom_value"
    assert expand_env_vars("${NONEXISTENT_VAR:-default_val}") == "default_val"
    assert expand_env_vars("postgresql://${RECQL_TEST_VAR}/db") == "postgresql://my_custom_value/db"


def test_multi_backend_catalog_parsing():
    yaml_doc = """
name: federated_engine
backends:
  postgres_main:
    backend: postgres
    dsn: postgresql://recql:recql@127.0.0.1:5432/recql
    min_size: 2
  oracle_emb:
    backend: oracle
    dsn: oracle://recql:recql@127.0.0.1:1521/FREEPDB1
  mariadb_text:
    backend: mariadb
    dsn: mysql://recql:recql@127.0.0.1:3306/recql

data:
  items:
    backend: postgres_main
    table: items
  users:
    backend: postgres_main
    table: users

index:
  embeddings:
    - name: content_embedding
      backend: postgres_main
      dims: 384
    - name: als
      backend: oracle_emb
      dims: 32

  lexical_search:
    backend: mariadb_text
    item_fields: [title, overview]

training:
  models:
    - name: ranker_lgbm
      backend: postgres_main
"""
    cat = load_engine_catalog(yaml_doc)
    assert cat.is_multi_backend() is True
    assert len(cat.backends) == 3
    assert cat.backends["postgres_main"].backend == "postgres"
    assert cat.backends["oracle_emb"].backend == "oracle"
    assert cat.backends["mariadb_text"].backend == "mariadb"

    assert cat.backend_for_embedding("content_embedding") == "postgres_main"
    assert cat.backend_for_embedding("als") == "oracle_emb"
    assert cat.backend_for_lexical() == "mariadb_text"
    assert cat.backend_for_model("ranker_lgbm") == "postgres_main"
    assert cat.backend_for_entity("item") == "postgres_main"

    bindings = cat.bindings()
    assert bindings.items.backend == "postgres_main"
    assert bindings.users.backend == "postgres_main"
    assert bindings.backend_for_embedding("content_embedding") == "postgres_main"
    assert bindings.backend_for_embedding("als") == "oracle_emb"
