"""Core dialect registry — no backend packs required."""

from __future__ import annotations

import pytest

from recql.errors import ExecuteError
from recql.plugins.dialect import get_sql_dialect


def test_unknown_backend_raises():
    with pytest.raises(ExecuteError, match="no SQL dialect"):
        get_sql_dialect("sqlite")
