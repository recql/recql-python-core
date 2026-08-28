"""Type helpers / Param specs (GraphQL-inspired coercion)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Param:
    name: str
    type: str = "Any"
    default: Any = None
    non_null: bool = False
