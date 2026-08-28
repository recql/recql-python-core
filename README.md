# recql-python-core

Storage-agnostic RecQL engine. No database drivers. Backend packs register via
entry points (`recql.backends` / `recql.dialects` / `recql.connectors`); core
does not hardcode backend names or aliases.

Engine YAML must set ``plugins.backend`` (no default).

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
