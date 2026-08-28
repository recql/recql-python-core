"""Lower + OpenAPI field name goldens."""

from __future__ import annotations

from recql.harness import recql_to_rank_query_config
from recql.language import parse
from recql.lower import lower_to_dict
from recql.yaml_query import load_rank_query_config, normalize_config_dict


def test_string_encoder_carries_input_user_id():
    sql = """
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder='interaction_pooling',
        input_user_id='$user_id',
        truncate_interactions=20,
        limit=50
      )
    ) LIMIT 10
    """
    step = lower_to_dict(parse(sql))["retrieve"][0]
    assert step["query_encoder"]["type"] == "interaction_pooling"
    assert step["query_encoder"]["input_user_id"] == "$user_id"
    assert step["query_encoder"]["truncate_interactions"] == 20


def test_vector_text_search_defaults_embedding_ref():
    sql = """
    SELECT * FROM retrieve(text_search(query='$query', mode='vector', limit=50))
    LIMIT 20
    """
    mode = lower_to_dict(parse(sql))["retrieve"][0]["mode"]
    assert mode["type"] == "vector"
    assert mode["text_embedding_ref"] == "content_embedding"


def test_trailing_comma_in_retrieve_call():
    sql = """
    SELECT * FROM retrieve(
      column_order(columns='_derived_popular_rank',)
    )
    """
    cfg = lower_to_dict(parse(sql))
    assert cfg["retrieve"][0]["type"] == "column_order"


def test_text_search_input_text_query():
    sql = """
    SELECT * FROM retrieve(
      text_search(query=$q, mode='vector', text_embedding_ref='content', name='vec', limit=30)
    ) LIMIT 10
    """
    cfg = lower_to_dict(parse(sql))
    step = cfg["retrieve"][0]
    assert step["type"] == "text_search"
    assert step["input_text_query"] == "$q"
    assert step["mode"]["type"] == "vector"
    assert step["mode"]["text_embedding_ref"] == "content"


def test_column_order_string_to_list():
    sql = "SELECT * FROM retrieve(column_order(columns='created_at DESC, price ASC', name='ord'))"
    cfg = lower_to_dict(parse(sql))
    cols = cfg["retrieve"][0]["columns"]
    assert cols[0] == {"name": "created_at", "ascending": False, "nulls_first": False}
    assert cols[1]["name"] == "price"
    assert cols[1]["ascending"] is True


def test_ids_to_candidate_ids():
    sql = "SELECT * FROM retrieve(ids(ids=['a','b','c'], name='cand'))"
    cfg = lower_to_dict(parse(sql))
    assert cfg["retrieve"][0]["type"] == "candidate_ids"
    assert cfg["retrieve"][0]["item_ids"] == ["a", "b", "c"]


def test_yaml_load_validates():
    raw = {
        "type": "rank",
        "from": "item",
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "als",
                "query_encoder": {
                    "type": "precomputed_user",
                    "input_user_id": "$parameter.user_id",
                },
                "limit": 50,
            }
        ],
        "limit": 10,
    }
    cfg = load_rank_query_config(raw)
    d = normalize_config_dict(raw)
    assert d["retrieve"][0]["query_encoder"]["type"] == "precomputed_user"


def test_recql_to_rank_query_config_parity():
    sql = """
    SELECT * FROM retrieve(
      similarity(
        embedding_ref='als',
        encoder=precomputed_user(input_user_id=$parameter.user_id),
        limit=50
      )
    ) LIMIT 10
    """
    yaml_obj = {
        "type": "rank",
        "from": "item",
        "retrieve": [
            {
                "type": "similarity",
                "embedding_ref": "als",
                "query_encoder": {
                    "type": "precomputed_user",
                    "input_user_id": "$parameter.user_id",
                },
                "limit": 50,
            }
        ],
        "limit": 10,
    }
    a = normalize_config_dict(recql_to_rank_query_config(sql))
    b = normalize_config_dict(recql_to_rank_query_config(yaml_obj))
    assert a["retrieve"][0]["type"] == b["retrieve"][0]["type"]
    assert a["retrieve"][0]["embedding_ref"] == b["retrieve"][0]["embedding_ref"]
    assert a["limit"] == b["limit"]


def test_order_by_reorder_alias_skips_column_sort():
    sql = """
    SELECT score(expression='click_through_rate', input_user_id=$user_id) AS s,
           boosted(
             score=s,
             retriever=filter(where='e.attrs @> ''{"genres":["Comedy"]}''::jsonb', limit=40),
             strength=0.35
           ) AS r, *
    FROM retrieve(
      similarity(embedding_ref='als', encoder=precomputed_user(input_user_id=$user_id), limit=100)
    )
    ORDER BY r
    LIMIT 20
    """
    cfg = lower_to_dict(parse(sql))
    types = [s["type"] for s in cfg.get("reorder") or []]
    assert "boosted" in types
    assert "column_sort" not in types
