"""RecQL errors with optional location."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SourceLocation:
    line: int
    column: int

    def __str__(self) -> str:
        return f"{self.line}:{self.column}"


class RecqlError(Exception):
    """Base RecQL error."""

    def __init__(
        self,
        message: str,
        *,
        locations: list[SourceLocation] | None = None,
        path: list[Any] | None = None,
        extensions: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.locations = locations or []
        self.path = path or []
        self.extensions = extensions or {}

    def __str__(self) -> str:
        if self.locations:
            loc = ", ".join(str(l) for l in self.locations)
            return f"{self.message} ({loc})"
        return self.message


class SyntaxError_(RecqlError):
    """Parse / lexer error (named to avoid shadowing builtins.SyntaxError)."""


class ValidationError(RecqlError):
    """Structural or semantic validation failure."""


class BindError(RecqlError):
    """Catalog / capability bind failure."""


class ExecuteError(RecqlError):
    """Runtime execution failure."""


class UnsupportedError(RecqlError):
    """Documented unsupported feature."""
