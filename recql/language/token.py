"""Lexer tokens for RecQL / SIDEQL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class TokenKind(Enum):
    EOF = auto()
    NAME = auto()
    COMPOUND_NAME = auto()  # a.b.c
    PARAM = auto()  # $name or $name.x
    STRING = auto()
    INTEGER = auto()
    FLOAT = auto()
    TRUE = auto()
    FALSE = auto()
    NULL = auto()

    # punctuation / ops
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    COMMA = auto()
    SEMICOLON = auto()
    DOT = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    PLUS = auto()
    MINUS = auto()
    STAR = auto()
    SLASH = auto()
    PERCENT = auto()
    POWER = auto()  # **


@dataclass(slots=True, frozen=True)
class Token:
    kind: TokenKind
    value: str | int | float | bool | None
    line: int
    column: int
    # For NAME / keywords: original lexeme (preserve case for identifiers)
    lexeme: str = ""


# Keywords matched case-insensitively → TokenKind where special, else NAME
KEYWORDS: dict[str, TokenKind] = {
    "true": TokenKind.TRUE,
    "false": TokenKind.FALSE,
    "null": TokenKind.NULL,
}
