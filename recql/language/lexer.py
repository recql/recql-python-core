"""RecQL lexer (Part 3.1 lexical structure)."""

from __future__ import annotations

from recql.errors import SourceLocation, SyntaxError_
from recql.language.token import KEYWORDS, Token, TokenKind


class Lexer:
    def __init__(self, source: str) -> None:
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1
        self._token: Token | None = None

    def peek(self) -> Token:
        if self._token is None:
            self._token = self._next()
        return self._token

    def advance(self) -> Token:
        tok = self.peek()
        self._token = None
        return tok

    def expect(self, kind: TokenKind) -> Token:
        tok = self.advance()
        if tok.kind != kind:
            raise SyntaxError_(
                f"expected {kind.name}, got {tok.kind.name}",
                locations=[SourceLocation(tok.line, tok.column)],
            )
        return tok

    def _next(self) -> Token:
        self._skip_ws_and_comments()
        if self.pos >= self.length:
            return Token(TokenKind.EOF, None, self.line, self.column)

        start_line, start_col = self.line, self.column
        ch = self.source[self.pos]

        # operators (multi-char first)
        if ch == "*" and self._peek_char(1) == "*":
            self._bump()
            self._bump()
            return Token(TokenKind.POWER, "**", start_line, start_col, "**")
        if ch == "!" and self._peek_char(1) == "=":
            self._bump()
            self._bump()
            return Token(TokenKind.NE, "!=", start_line, start_col, "!=")
        if ch == "<" and self._peek_char(1) == ">":
            self._bump()
            self._bump()
            return Token(TokenKind.NE, "<>", start_line, start_col, "<>")
        if ch == "<" and self._peek_char(1) == "=":
            self._bump()
            self._bump()
            return Token(TokenKind.LE, "<=", start_line, start_col, "<=")
        if ch == ">" and self._peek_char(1) == "=":
            self._bump()
            self._bump()
            return Token(TokenKind.GE, ">=", start_line, start_col, ">=")

        single = {
            "(": TokenKind.LPAREN,
            ")": TokenKind.RPAREN,
            "[": TokenKind.LBRACKET,
            "]": TokenKind.RBRACKET,
            ",": TokenKind.COMMA,
            ";": TokenKind.SEMICOLON,
            "=": TokenKind.EQ,
            "<": TokenKind.LT,
            ">": TokenKind.GT,
            "+": TokenKind.PLUS,
            "-": TokenKind.MINUS,
            "*": TokenKind.STAR,
            "/": TokenKind.SLASH,
            "%": TokenKind.PERCENT,
        }
        if ch in single:
            self._bump()
            return Token(single[ch], ch, start_line, start_col, ch)

        if ch in ("'", '"'):
            return self._string(ch, start_line, start_col)

        if ch == "$":
            return self._param(start_line, start_col)

        if ch.isdigit() or (ch == "." and self._peek_char(1).isdigit()):
            return self._number(start_line, start_col)

        if ch.isalpha() or ch == "_":
            return self._name_or_keyword(start_line, start_col)

        raise SyntaxError_(
            f"unexpected character {ch!r}",
            locations=[SourceLocation(start_line, start_col)],
        )

    def _skip_ws_and_comments(self) -> None:
        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch in " \t\r\n":
                self._bump()
                continue
            if ch == "-" and self._peek_char(1) == "-":
                while self.pos < self.length and self.source[self.pos] != "\n":
                    self._bump()
                continue
            break

    def _string(self, quote: str, line: int, col: int) -> Token:
        self._bump()  # opening
        chars: list[str] = []
        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch == quote:
                # doubled delimiter escape
                if self._peek_char(1) == quote:
                    self._bump()
                    self._bump()
                    chars.append(quote)
                    continue
                self._bump()
                return Token(TokenKind.STRING, "".join(chars), line, col)
            if ch == "\\" and self._peek_char(1) == "\n":
                # Docs sometimes split long score expressions with backslash-newline.
                self._bump()
                self._bump()
                continue
            if ch == "\\" and self._peek_char(1) == "\r" and self._peek_char(2) == "\n":
                self._bump()
                self._bump()
                self._bump()
                continue
            if ch == "\n":
                raise SyntaxError_(
                    "unterminated string",
                    locations=[SourceLocation(line, col)],
                )
            chars.append(ch)
            self._bump()
        raise SyntaxError_("unterminated string", locations=[SourceLocation(line, col)])

    def _param(self, line: int, col: int) -> Token:
        self._bump()  # $
        if self.pos >= self.length or not (
            self.source[self.pos].isalpha() or self.source[self.pos] == "_"
        ):
            raise SyntaxError_(
                "expected parameter name after $",
                locations=[SourceLocation(line, col)],
            )
        parts = [self._read_ident()]
        while self.pos < self.length and self.source[self.pos] == ".":
            # peek if next is ident
            if self.pos + 1 < self.length and (
                self.source[self.pos + 1].isalpha() or self.source[self.pos + 1] == "_"
            ):
                self._bump()  # .
                parts.append(self._read_ident())
            else:
                break
        value = "$" + ".".join(parts)
        return Token(TokenKind.PARAM, value, line, col, value)

    def _number(self, line: int, col: int) -> Token:
        start = self.pos
        is_float = False
        while self.pos < self.length and self.source[self.pos].isdigit():
            self._bump()
        if self.pos < self.length and self.source[self.pos] == ".":
            is_float = True
            self._bump()
            while self.pos < self.length and self.source[self.pos].isdigit():
                self._bump()
        if self.pos < self.length and self.source[self.pos] in "eE":
            is_float = True
            self._bump()
            if self.pos < self.length and self.source[self.pos] in "+-":
                self._bump()
            if self.pos >= self.length or not self.source[self.pos].isdigit():
                raise SyntaxError_(
                    "invalid float literal",
                    locations=[SourceLocation(line, col)],
                )
            while self.pos < self.length and self.source[self.pos].isdigit():
                self._bump()
        lexeme = self.source[start : self.pos]
        # fix column accounting — _bump already advanced line/col
        if is_float:
            return Token(TokenKind.FLOAT, float(lexeme), line, col, lexeme)
        return Token(TokenKind.INTEGER, int(lexeme), line, col, lexeme)

    def _name_or_keyword(self, line: int, col: int) -> Token:
        parts = [self._read_ident()]
        # compound name a.b.c — but NOT if `.` starts a float (handled) or
        # followed by non-ident. Keywords only for single segment.
        while self.pos < self.length and self.source[self.pos] == ".":
            nxt = self._peek_char(1)
            if nxt.isalpha() or nxt == "_":
                self._bump()
                parts.append(self._read_ident())
            else:
                break
        if len(parts) == 1:
            lower = parts[0].lower()
            if lower in KEYWORDS:
                kind = KEYWORDS[lower]
                val: bool | None
                if kind is TokenKind.TRUE:
                    val = True
                elif kind is TokenKind.FALSE:
                    val = False
                else:
                    val = None
                return Token(kind, val, line, col, parts[0])
            return Token(TokenKind.NAME, parts[0], line, col, parts[0])
        compound = ".".join(parts)
        return Token(TokenKind.COMPOUND_NAME, compound, line, col, compound)

    def _read_ident(self) -> str:
        start = self.pos
        while self.pos < self.length:
            ch = self.source[self.pos]
            if ch.isalnum() or ch in "_$":
                self._bump()
            else:
                break
        return self.source[start : self.pos]

    def _peek_char(self, offset: int = 0) -> str:
        i = self.pos + offset
        if i >= self.length:
            return ""
        return self.source[i]

    def _bump(self) -> None:
        if self.pos < self.length:
            if self.source[self.pos] == "\n":
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.pos += 1
