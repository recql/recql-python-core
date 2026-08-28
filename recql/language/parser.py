"""RecQL recursive-descent parser implementing Part 3 EBNF."""

from __future__ import annotations

from dataclasses import dataclass

from recql.errors import SourceLocation, SyntaxError_, UnsupportedError
from recql.language import ast as A
from recql.language.lexer import Lexer
from recql.language.token import Token, TokenKind


@dataclass
class FeatureFlags:
    """Parse-time flags for *legacy SIDEQL forms the docs still show*.

    Analytics SQL (WITH/CTE, UNION, JOIN, …) is **not** in the RecQL grammar.
    Those keywords are rejected at the token — there is no production to parse
    them (O-R8). Do not add allow_* flags that revive C-EBNF extensions.
    """

    allow_legacy_from: bool = True
    allow_order_by_score: bool = True
    allow_reorder_by: bool = True
    allow_case: bool = True


# Keywords that are not RecQL/SIDEQL. Seen → error; never parsed into AST.
_NOT_IN_LANGUAGE = {
    "WITH": "Common Table Expressions (WITH clause) are not supported",
    "JOIN": "JOINs are not supported",
    "GROUP": "GROUP BY / aggregations are not supported",
    "HAVING": "HAVING clause is not supported",
    "OVER": "Window functions (OVER clause) are not supported",
    "WINDOW": "Window functions are not supported",
    "INTERSECT": "INTERSECT is not supported",
    "EXCEPT": "EXCEPT is not supported",
    "UNION": "UNION is not supported",
}


class Parser:
    def __init__(self, source: str, *, flags: FeatureFlags | None = None) -> None:
        self.lex = Lexer(source)
        self.flags = flags or FeatureFlags()

    def parse_query(self) -> A.SelectStmt:
        stmt = self.parse_select_stmt()
        if self.lex.peek().kind is TokenKind.SEMICOLON:
            self.lex.advance()
            if self.lex.peek().kind is not TokenKind.EOF:
                tok = self.lex.peek()
                raise SyntaxError_(
                    "multiple statements are not supported",
                    locations=[SourceLocation(tok.line, tok.column)],
                )
        self.lex.expect(TokenKind.EOF)
        self._validate_stmt(stmt)
        return stmt

    def parse_expr_only(self) -> A.Expr:
        expr = self.parse_expr()
        self.lex.expect(TokenKind.EOF)
        return expr

    # ------------------------------------------------------------------ select
    def parse_select_stmt(self) -> A.SelectStmt:
        self._reject_if_not_in_language(self.lex.peek())

        sel = self.lex.peek()
        if not self._is_kw(sel, "SELECT"):
            raise SyntaxError_(
                "expected SELECT",
                locations=[SourceLocation(sel.line, sel.column)],
            )
        self.lex.advance()
        select_list = self._parse_select_list()

        from_sources: list[A.FromSource] = []
        if self._is_kw(self.lex.peek(), "FROM"):
            self.lex.advance()
            from_sources = self._parse_from_clause()

        self._reject_if_not_in_language(self.lex.peek())

        where = None
        if self._is_kw(self.lex.peek(), "WHERE"):
            self.lex.advance()
            where = self.parse_expr()

        order_by: list[A.OrderItem] = []
        if self._is_kw(self.lex.peek(), "ORDER"):
            self.lex.advance()
            self._expect_kw("BY")
            order_by = self._parse_order_list()

        reorder_by: list[A.FuncCall] = []
        if self._is_kw(self.lex.peek(), "REORDER"):
            tok = self.lex.peek()
            if not self.flags.allow_reorder_by:
                raise UnsupportedError(
                    "legacy syntax disabled: REORDER BY",
                    locations=[SourceLocation(tok.line, tok.column)],
                )
            self.lex.advance()
            self._expect_kw("BY")
            reorder_by = self._parse_reorder_list()

        limit: int | A.Param | None = None
        if self._is_kw(self.lex.peek(), "LIMIT"):
            self.lex.advance()
            limit = self._parse_limit_value()

        offset: int | A.Param | None = None
        if self._is_kw(self.lex.peek(), "OFFSET"):
            self.lex.advance()
            offset = self._parse_limit_value()

        return A.SelectStmt(
            line=sel.line,
            column=sel.column,
            select_list=select_list,
            from_sources=from_sources,
            where=where,
            order_by=order_by,
            reorder_by=reorder_by,
            limit=limit,
            offset=offset,
        )

    def _reject_if_not_in_language(self, tok: Token) -> None:
        """Error on keywords that are not RecQL — no grammar production exists."""
        if tok.kind not in (TokenKind.NAME, TokenKind.COMPOUND_NAME):
            return
        if not isinstance(tok.value, str):
            return
        upper = tok.value.upper()
        if upper.startswith("GROUP"):
            upper = "GROUP"
        # JOIN can be preceded by LEFT/RIGHT/INNER/FULL/CROSS
        if upper in ("LEFT", "RIGHT", "INNER", "FULL", "CROSS"):
            raise UnsupportedError(
                _NOT_IN_LANGUAGE["JOIN"],
                locations=[SourceLocation(tok.line, tok.column)],
            )
        msg = _NOT_IN_LANGUAGE.get(upper)
        if msg is not None:
            raise UnsupportedError(
                msg, locations=[SourceLocation(tok.line, tok.column)]
            )

    def _parse_select_list(self) -> list[A.SelectItem]:
        items = [self._parse_select_item()]
        while self.lex.peek().kind is TokenKind.COMMA:
            self.lex.advance()
            items.append(self._parse_select_item())
        return items

    def _parse_select_item(self) -> A.SelectItem:
        tok = self.lex.peek()
        if tok.kind is TokenKind.STAR:
            self.lex.advance()
            if self._is_kw(self.lex.peek(), "AS"):
                raise SyntaxError_(
                    "SELECT * must not take AS alias",
                    locations=[SourceLocation(tok.line, tok.column)],
                )
            return A.SelectItem(line=tok.line, column=tok.column, star=True)
        expr = self.parse_expr()
        alias = None
        if self._is_kw(self.lex.peek(), "AS"):
            self.lex.advance()
            alias = self._expect_name()
        return A.SelectItem(
            line=tok.line, column=tok.column, expr=expr, alias=alias
        )

    def _parse_from_clause(self) -> list[A.FromSource]:
        sources = [self._parse_from_source()]
        while self.lex.peek().kind is TokenKind.COMMA:
            if not self.flags.allow_legacy_from:
                tok = self.lex.peek()
                raise UnsupportedError(
                    "legacy syntax disabled: multi-source FROM",
                    locations=[SourceLocation(tok.line, tok.column)],
                )
            self.lex.advance()
            sources.append(self._parse_from_source())
        return sources

    def _parse_from_source(self) -> A.FromSource:
        tok = self.lex.peek()
        self._reject_if_not_in_language(tok)

        # Parenthesized FROM subquery is not RecQL (docs: no subqueries in FROM).
        if tok.kind is TokenKind.LPAREN:
            raise UnsupportedError(
                "Subqueries in FROM clause are not supported",
                locations=[SourceLocation(tok.line, tok.column)],
            )

        # engine.<name>.retrieve(...) or engine.<name>.items etc.
        if self._is_kw(tok, "ENGINE") or (
            tok.kind is TokenKind.COMPOUND_NAME
            and isinstance(tok.value, str)
            and tok.value.lower().startswith("engine.")
        ):
            return self._parse_engine_or_table()

        if self._is_kw(tok, "DATA") or (
            tok.kind is TokenKind.COMPOUND_NAME
            and isinstance(tok.value, str)
            and tok.value.lower().startswith("data.")
        ):
            return self._parse_table_ref()

        if self._is_kw(tok, "RETRIEVE"):
            return self._parse_retrieve_call(engine_name=None)

        if self._is_kw(tok, "ITEMS") or self._is_kw(tok, "USERS"):
            name = self._expect_name()
            return A.TableRef(line=tok.line, column=tok.column, path=[name.lower()])

        # bare name — could be legacy bare retriever call or table/CTE ref
        if tok.kind in (TokenKind.NAME, TokenKind.COMPOUND_NAME):
            # Lookahead: name '(' → func call (legacy bare retriever)
            name_tok = self.lex.advance()
            if self.lex.peek().kind is TokenKind.LPAREN:
                if not self.flags.allow_legacy_from:
                    raise UnsupportedError(
                        "legacy syntax disabled: bare retriever in FROM",
                        locations=[SourceLocation(name_tok.line, name_tok.column)],
                    )
                call = self._finish_func_call(name_tok)
                return A.RetrieveCall(
                    line=name_tok.line,
                    column=name_tok.column,
                    calls=[call],
                    bare=True,
                )
            # table / CTE ref — may continue with .parts if NAME only
            path = [str(name_tok.value)]
            if name_tok.kind is TokenKind.NAME:
                while self.lex.peek().kind is TokenKind.DOT:
                    self.lex.advance()
                    path.append(self._expect_name())
            return A.TableRef(
                line=name_tok.line, column=name_tok.column, path=path
            )

        raise SyntaxError_(
            "expected FROM source",
            locations=[SourceLocation(tok.line, tok.column)],
        )

    def _parse_engine_or_table(self) -> A.FromSource:
        tok = self.lex.peek()
        if tok.kind is TokenKind.COMPOUND_NAME:
            parts = str(tok.value).split(".")
            self.lex.advance()
        else:
            # ENGINE . name . ...
            self.lex.advance()  # ENGINE
            self.lex.expect(TokenKind.DOT)
            parts = ["engine", self._expect_name()]
            while self.lex.peek().kind is TokenKind.DOT:
                self.lex.advance()
                parts.append(self._expect_name())

        lower = [p.lower() for p in parts]
        if len(lower) >= 3 and lower[0] == "engine" and lower[2] == "retrieve":
            engine_name = parts[1]
            if self.lex.peek().kind is TokenKind.LPAREN:
                return self._parse_retrieve_call(engine_name=engine_name, already_named=True)
            raise SyntaxError_(
                "expected ( after retrieve",
                locations=[SourceLocation(tok.line, tok.column)],
            )
        return A.TableRef(line=tok.line, column=tok.column, path=parts)

    def _parse_table_ref(self) -> A.TableRef:
        tok = self.lex.peek()
        if tok.kind is TokenKind.COMPOUND_NAME:
            path = str(tok.value).split(".")
            self.lex.advance()
            return A.TableRef(line=tok.line, column=tok.column, path=path)
        self.lex.advance()  # DATA
        path = ["data"]
        while self.lex.peek().kind is TokenKind.DOT:
            self.lex.advance()
            path.append(self._expect_name())
        return A.TableRef(line=tok.line, column=tok.column, path=path)

    def _parse_retrieve_call(
        self, *, engine_name: str | None, already_named: bool = False
    ) -> A.RetrieveCall:
        if not already_named:
            tok = self.lex.peek()
            self._expect_kw("RETRIEVE")
        else:
            tok = self.lex.peek()
        self.lex.expect(TokenKind.LPAREN)
        calls: list[A.FuncCall] = []
        if self.lex.peek().kind is not TokenKind.RPAREN:
            calls.append(self._parse_func_call())
            while self.lex.peek().kind is TokenKind.COMMA:
                self.lex.advance()
                calls.append(self._parse_func_call())
        self.lex.expect(TokenKind.RPAREN)
        return A.RetrieveCall(
            line=tok.line,
            column=tok.column,
            engine_name=engine_name,
            calls=calls,
            bare=False,
        )

    def _parse_order_list(self) -> list[A.OrderItem]:
        items = [self._parse_order_item()]
        while self.lex.peek().kind is TokenKind.COMMA:
            self.lex.advance()
            items.append(self._parse_order_item())
        return items

    def _parse_order_item(self) -> A.OrderItem:
        tok = self.lex.peek()
        key: A.Name | A.FuncCall
        if tok.kind in (TokenKind.NAME, TokenKind.COMPOUND_NAME):
            name_tok = self.lex.advance()
            if self.lex.peek().kind is TokenKind.LPAREN:
                if not self.flags.allow_order_by_score:
                    raise UnsupportedError(
                        "legacy syntax disabled: ORDER BY score(...)",
                        locations=[SourceLocation(name_tok.line, name_tok.column)],
                    )
                key = self._finish_func_call(name_tok)
            else:
                key = A.Name(
                    line=name_tok.line,
                    column=name_tok.column,
                    value=str(name_tok.value),
                )
        else:
            raise SyntaxError_(
                "expected ORDER BY key",
                locations=[SourceLocation(tok.line, tok.column)],
            )
        direction = "ASC"
        if self._is_kw(self.lex.peek(), "ASC"):
            self.lex.advance()
            direction = "ASC"
        elif self._is_kw(self.lex.peek(), "DESC"):
            self.lex.advance()
            direction = "DESC"
        nulls = None
        if self._is_kw(self.lex.peek(), "NULLS"):
            self.lex.advance()
            if self._is_kw(self.lex.peek(), "FIRST"):
                self.lex.advance()
                nulls = "FIRST"
            elif self._is_kw(self.lex.peek(), "LAST"):
                self.lex.advance()
                nulls = "LAST"
            else:
                t = self.lex.peek()
                raise SyntaxError_(
                    "expected NULLS FIRST|LAST",
                    locations=[SourceLocation(t.line, t.column)],
                )
        return A.OrderItem(
            line=tok.line, column=tok.column, key=key, direction=direction, nulls=nulls
        )

    def _parse_reorder_list(self) -> list[A.FuncCall]:
        calls = [self._parse_func_call()]
        while self.lex.peek().kind is TokenKind.COMMA:
            self.lex.advance()
            calls.append(self._parse_func_call())
        return calls

    def _parse_limit_value(self) -> int | A.Param:
        tok = self.lex.peek()
        if tok.kind is TokenKind.INTEGER:
            self.lex.advance()
            val = int(tok.value)  # type: ignore[arg-type]
            if val < 0:
                raise SyntaxError_(
                    "LIMIT/OFFSET must be non-negative",
                    locations=[SourceLocation(tok.line, tok.column)],
                )
            return val
        if tok.kind is TokenKind.PARAM:
            self.lex.advance()
            return A.Param(line=tok.line, column=tok.column, value=str(tok.value))
        raise SyntaxError_(
            "expected integer or parameter for LIMIT/OFFSET",
            locations=[SourceLocation(tok.line, tok.column)],
        )

    # ------------------------------------------------------------------ calls
    def _parse_func_call(self) -> A.FuncCall:
        tok = self.lex.peek()
        if tok.kind not in (TokenKind.NAME, TokenKind.COMPOUND_NAME):
            raise SyntaxError_(
                "expected function name",
                locations=[SourceLocation(tok.line, tok.column)],
            )
        name_tok = self.lex.advance()
        return self._finish_func_call(name_tok)

    def _finish_func_call(self, name_tok: Token) -> A.FuncCall:
        self.lex.expect(TokenKind.LPAREN)
        args: list[A.CallArg] = []
        if self.lex.peek().kind is not TokenKind.RPAREN:
            args.append(self._parse_call_arg())
            while self.lex.peek().kind is TokenKind.COMMA:
                self.lex.advance()
                # Trailing comma before ')' is allowed (docs often use it).
                if self.lex.peek().kind is TokenKind.RPAREN:
                    break
                args.append(self._parse_call_arg())
        self.lex.expect(TokenKind.RPAREN)
        return A.FuncCall(
            line=name_tok.line,
            column=name_tok.column,
            name=str(name_tok.value),
            args=args,
        )

    def _parse_call_arg(self) -> A.CallArg:
        """D-15: at start of call_arg, name '=' is always the arg name."""
        tok = self.lex.peek()
        if tok.kind is TokenKind.NAME:
            # Lookahead for '='
            saved = self.lex
            # Manual lookahead without consuming permanently if not named
            name_tok = self.lex.advance()
            if self.lex.peek().kind is TokenKind.EQ:
                self.lex.advance()
                value = self.parse_expr()
                return A.CallArg(
                    line=name_tok.line,
                    column=name_tok.column,
                    name=str(name_tok.value),
                    value=value,
                )
            # Not named — name is start of expr; push back by re-parsing
            # We already consumed NAME; treat as primary name in expr.
            # Build expr starting from this name.
            expr = self._expr_from_consumed_name(name_tok)
            return A.CallArg(line=tok.line, column=tok.column, value=expr)
        value = self.parse_expr()
        return A.CallArg(line=tok.line, column=tok.column, value=value)

    def _expr_from_consumed_name(self, name_tok: Token) -> A.Expr:
        """Continue parsing an expression after a NAME token was consumed."""
        if self.lex.peek().kind is TokenKind.LPAREN:
            primary: A.Expr = self._finish_func_call(name_tok)
        else:
            primary = A.Name(
                line=name_tok.line, column=name_tok.column, value=str(name_tok.value)
            )
        return self._parse_expr_from_primary(primary)

    # ------------------------------------------------------------------ expr
    def parse_expr(self) -> A.Expr:
        return self._parse_or()

    def _parse_expr_from_primary(self, primary: A.Expr) -> A.Expr:
        """Resume layered expr parse after a primary was already built."""
        # pow / mul / add / pred / not / and / or — primary is already past unary
        left = self._continue_pow(primary)
        left = self._continue_mul(left)
        left = self._continue_add(left)
        left = self._continue_pred(left)
        # not_expr is only prefix NOT — skip
        left = self._continue_and(left)
        left = self._continue_or(left)
        return left

    def _parse_or(self) -> A.Expr:
        left = self._parse_and()
        return self._continue_or(left)

    def _continue_or(self, left: A.Expr) -> A.Expr:
        while self._is_kw(self.lex.peek(), "OR"):
            op = self.lex.advance()
            right = self._parse_and()
            left = A.BinaryOp(
                line=op.line, column=op.column, op="OR", left=left, right=right
            )
        return left

    def _parse_and(self) -> A.Expr:
        left = self._parse_not()
        return self._continue_and(left)

    def _continue_and(self, left: A.Expr) -> A.Expr:
        while self._is_kw(self.lex.peek(), "AND"):
            op = self.lex.advance()
            right = self._parse_not()
            left = A.BinaryOp(
                line=op.line, column=op.column, op="AND", left=left, right=right
            )
        return left

    def _parse_not(self) -> A.Expr:
        if self._is_kw(self.lex.peek(), "NOT"):
            op = self.lex.advance()
            return A.UnaryOp(
                line=op.line, column=op.column, op="NOT", operand=self._parse_not()
            )
        return self._parse_pred()

    def _parse_pred(self) -> A.Expr:
        left = self._parse_add()
        return self._continue_pred(left)

    def _continue_pred(self, left: A.Expr) -> A.Expr:
        while True:
            tok = self.lex.peek()
            if tok.kind in (
                TokenKind.EQ,
                TokenKind.NE,
                TokenKind.LT,
                TokenKind.LE,
                TokenKind.GT,
                TokenKind.GE,
            ):
                op_tok = self.lex.advance()
                right = self._parse_add()
                left = A.BinaryOp(
                    line=op_tok.line,
                    column=op_tok.column,
                    op=str(op_tok.lexeme or op_tok.value),
                    left=left,
                    right=right,
                )
                continue
            if self._is_kw(tok, "IS"):
                self.lex.advance()
                negated = False
                if self._is_kw(self.lex.peek(), "NOT"):
                    self.lex.advance()
                    negated = True
                target_tok = self.lex.peek()
                if self._is_kw(target_tok, "NULL"):
                    self.lex.advance()
                    target = "NULL"
                elif self._is_kw(target_tok, "TRUE"):
                    self.lex.advance()
                    target = "TRUE"
                elif self._is_kw(target_tok, "FALSE"):
                    self.lex.advance()
                    target = "FALSE"
                else:
                    raise SyntaxError_(
                        "expected NULL|TRUE|FALSE after IS",
                        locations=[SourceLocation(target_tok.line, target_tok.column)],
                    )
                left = A.IsPredicate(
                    line=tok.line,
                    column=tok.column,
                    expr=left,
                    negated=negated,
                    target=target,
                )
                continue
            # NOT IN / IN / LIKE / ILIKE / BETWEEN
            negated = False
            if self._is_kw(tok, "NOT"):
                # peek further
                self.lex.advance()
                nxt = self.lex.peek()
                if self._is_kw(nxt, "IN"):
                    negated = True
                    left = self._parse_in(left, negated=True, loc=tok)
                    continue
                if self._is_kw(nxt, "LIKE") or self._is_kw(nxt, "ILIKE"):
                    negated = True
                    left = self._parse_like(left, negated=True, loc=tok)
                    continue
                if self._is_kw(nxt, "BETWEEN"):
                    left = self._parse_between(left, negated=True, loc=tok)
                    continue
                raise SyntaxError_(
                    "unexpected NOT in predicate",
                    locations=[SourceLocation(tok.line, tok.column)],
                )
            if self._is_kw(tok, "IN"):
                left = self._parse_in(left, negated=False, loc=tok)
                continue
            if self._is_kw(tok, "LIKE") or self._is_kw(tok, "ILIKE"):
                left = self._parse_like(left, negated=False, loc=tok)
                continue
            if self._is_kw(tok, "BETWEEN"):
                left = self._parse_between(left, negated=False, loc=tok)
                continue
            break
        return left

    def _parse_in(self, left: A.Expr, *, negated: bool, loc: Token) -> A.InPredicate:
        self._expect_kw("IN")
        tok = self.lex.peek()
        if tok.kind is TokenKind.PARAM:
            self.lex.advance()
            return A.InPredicate(
                line=loc.line,
                column=loc.column,
                expr=left,
                negated=negated,
                param=A.Param(line=tok.line, column=tok.column, value=str(tok.value)),
            )
        if tok.kind is TokenKind.LPAREN:
            self.lex.advance()
            values: list[A.Expr] = []
            if self.lex.peek().kind is not TokenKind.RPAREN:
                values.append(self.parse_expr())
                while self.lex.peek().kind is TokenKind.COMMA:
                    self.lex.advance()
                    values.append(self.parse_expr())
            self.lex.expect(TokenKind.RPAREN)
            return A.InPredicate(
                line=loc.line,
                column=loc.column,
                expr=left,
                negated=negated,
                values=values,
            )
        if tok.kind is TokenKind.LBRACKET:
            arr = self._parse_array()
            return A.InPredicate(
                line=loc.line,
                column=loc.column,
                expr=left,
                negated=negated,
                values=list(arr.elements),
            )
        raise SyntaxError_(
            "expected (, [, or $param after IN",
            locations=[SourceLocation(tok.line, tok.column)],
        )

    def _parse_like(self, left: A.Expr, *, negated: bool, loc: Token) -> A.LikePredicate:
        tok = self.lex.peek()
        ci = self._is_kw(tok, "ILIKE")
        self.lex.advance()
        pattern = self._parse_add()
        return A.LikePredicate(
            line=loc.line,
            column=loc.column,
            expr=left,
            negated=negated,
            case_insensitive=ci,
            pattern=pattern,
        )

    def _parse_between(
        self, left: A.Expr, *, negated: bool, loc: Token
    ) -> A.BetweenPredicate:
        self._expect_kw("BETWEEN")
        # Narrowed context: low AND high — AND here is not boolean AND
        low = self._parse_add()
        self._expect_kw("AND")
        high = self._parse_add()
        return A.BetweenPredicate(
            line=loc.line,
            column=loc.column,
            expr=left,
            negated=negated,
            low=low,
            high=high,
        )

    def _parse_add(self) -> A.Expr:
        left = self._parse_mul()
        return self._continue_add(left)

    def _continue_add(self, left: A.Expr) -> A.Expr:
        while self.lex.peek().kind in (TokenKind.PLUS, TokenKind.MINUS):
            op = self.lex.advance()
            right = self._parse_mul()
            left = A.BinaryOp(
                line=op.line,
                column=op.column,
                op=str(op.lexeme),
                left=left,
                right=right,
            )
        return left

    def _parse_mul(self) -> A.Expr:
        left = self._parse_pow()
        return self._continue_mul(left)

    def _continue_mul(self, left: A.Expr) -> A.Expr:
        while self.lex.peek().kind in (
            TokenKind.STAR,
            TokenKind.SLASH,
            TokenKind.PERCENT,
        ):
            op = self.lex.advance()
            right = self._parse_pow()
            left = A.BinaryOp(
                line=op.line,
                column=op.column,
                op=str(op.lexeme),
                left=left,
                right=right,
            )
        return left

    def _parse_pow(self) -> A.Expr:
        # right-associative
        base = self._parse_unary()
        return self._continue_pow(base)

    def _continue_pow(self, base: A.Expr) -> A.Expr:
        if self.lex.peek().kind is TokenKind.POWER:
            op = self.lex.advance()
            # right-assoc: parse full pow on RHS
            right = self._parse_pow()
            return A.BinaryOp(
                line=op.line, column=op.column, op="**", left=base, right=right
            )
        return base

    def _parse_unary(self) -> A.Expr:
        if self.lex.peek().kind is TokenKind.MINUS:
            op = self.lex.advance()
            return A.UnaryOp(
                line=op.line, column=op.column, op="-", operand=self._parse_unary()
            )
        return self._parse_primary()

    def _parse_primary(self) -> A.Expr:
        tok = self.lex.peek()
        if tok.kind is TokenKind.LPAREN:
            self.lex.advance()
            expr = self.parse_expr()
            self.lex.expect(TokenKind.RPAREN)
            return expr
        if tok.kind is TokenKind.LBRACKET:
            return self._parse_array()
        if tok.kind is TokenKind.STRING:
            self.lex.advance()
            return A.Literal(
                line=tok.line, column=tok.column, value=tok.value, kind="string"
            )
        if tok.kind is TokenKind.INTEGER:
            self.lex.advance()
            return A.Literal(
                line=tok.line, column=tok.column, value=tok.value, kind="int"
            )
        if tok.kind is TokenKind.FLOAT:
            self.lex.advance()
            return A.Literal(
                line=tok.line, column=tok.column, value=tok.value, kind="float"
            )
        if tok.kind is TokenKind.TRUE:
            self.lex.advance()
            return A.Literal(line=tok.line, column=tok.column, value=True, kind="bool")
        if tok.kind is TokenKind.FALSE:
            self.lex.advance()
            return A.Literal(
                line=tok.line, column=tok.column, value=False, kind="bool"
            )
        if tok.kind is TokenKind.NULL:
            self.lex.advance()
            return A.Literal(line=tok.line, column=tok.column, value=None, kind="null")
        if tok.kind is TokenKind.PARAM:
            self.lex.advance()
            return A.Param(line=tok.line, column=tok.column, value=str(tok.value))
        if self._is_kw(tok, "CAST"):
            return self._parse_cast()
        if self._is_kw(tok, "CASE"):
            if not self.flags.allow_case:
                raise UnsupportedError(
                    "CASE is not supported in RecQL/SIDEQL with allow_case=OFF",
                    locations=[SourceLocation(tok.line, tok.column)],
                )
            return self._parse_case()
        if tok.kind in (TokenKind.NAME, TokenKind.COMPOUND_NAME):
            name_tok = self.lex.advance()
            if self.lex.peek().kind is TokenKind.LPAREN:
                return self._finish_func_call(name_tok)
            return A.Name(
                line=name_tok.line,
                column=name_tok.column,
                value=str(name_tok.value),
            )
        raise SyntaxError_(
            f"unexpected token {tok.kind.name}",
            locations=[SourceLocation(tok.line, tok.column)],
        )

    def _parse_array(self) -> A.ArrayLiteral:
        tok = self.lex.expect(TokenKind.LBRACKET)
        elements: list[A.Expr] = []
        if self.lex.peek().kind is not TokenKind.RBRACKET:
            elements.append(self.parse_expr())
            while self.lex.peek().kind is TokenKind.COMMA:
                self.lex.advance()
                elements.append(self.parse_expr())
        self.lex.expect(TokenKind.RBRACKET)
        return A.ArrayLiteral(line=tok.line, column=tok.column, elements=elements)

    def _parse_cast(self) -> A.CastExpr:
        tok = self.lex.advance()  # CAST
        self.lex.expect(TokenKind.LPAREN)
        expr = self.parse_expr()
        self._expect_kw("AS")
        type_name = self._expect_name()
        self.lex.expect(TokenKind.RPAREN)
        return A.CastExpr(
            line=tok.line, column=tok.column, expr=expr, type_name=type_name
        )

    def _parse_case(self) -> A.CaseExpr:
        tok = self.lex.advance()  # CASE
        operand = None
        if not self._is_kw(self.lex.peek(), "WHEN"):
            operand = self.parse_expr()
        whens: list[A.CaseWhen] = []
        while self._is_kw(self.lex.peek(), "WHEN"):
            wtok = self.lex.advance()
            cond = self.parse_expr()
            self._expect_kw("THEN")
            result = self.parse_expr()
            whens.append(
                A.CaseWhen(
                    line=wtok.line, column=wtok.column, condition=cond, result=result
                )
            )
        else_result = None
        if self._is_kw(self.lex.peek(), "ELSE"):
            self.lex.advance()
            else_result = self.parse_expr()
        self._expect_kw("END")
        return A.CaseExpr(
            line=tok.line,
            column=tok.column,
            operand=operand,
            whens=whens,
            else_result=else_result,
        )

    # ------------------------------------------------------------------ utils
    def _validate_stmt(self, stmt: A.SelectStmt) -> None:
        score_count = 0
        for item in stmt.select_list:
            if item.star:
                continue
            if isinstance(item.expr, A.FuncCall) and item.expr.name.lower() == "score":
                score_count += 1
                if item.alias is None:
                    raise SyntaxError_(
                        "score(...) requires AS alias",
                        locations=[SourceLocation(item.line, item.column)],
                    )
            elif isinstance(item.expr, A.FuncCall) and item.expr.name.lower() in (
                "diversity",
                "exploration",
                "boosted",
            ):
                if item.alias is None:
                    raise SyntaxError_(
                        f"{item.expr.name}(...) requires AS alias",
                        locations=[SourceLocation(item.line, item.column)],
                    )
        for oi in stmt.order_by:
            if isinstance(oi.key, A.FuncCall) and oi.key.name.lower() == "score":
                score_count += 1
        if score_count > 1:
            raise SyntaxError_(
                "at most one score() per query",
                locations=[SourceLocation(stmt.line, stmt.column)],
            )
        # data.* vs engine.* mix
        has_data = False
        has_engine = False
        for src in stmt.from_sources:
            if isinstance(src, A.TableRef):
                if src.path and src.path[0].lower() == "data":
                    has_data = True
                if src.path and src.path[0].lower() == "engine":
                    has_engine = True
            if isinstance(src, A.RetrieveCall) and src.engine_name:
                has_engine = True
        if has_data and has_engine:
            raise SyntaxError_(
                "data.* and engine.* namespaces must not mix in one query",
                locations=[SourceLocation(stmt.line, stmt.column)],
            )

    def _expect_name(self) -> str:
        tok = self.lex.peek()
        if tok.kind is TokenKind.NAME:
            self.lex.advance()
            return str(tok.value)
        # keywords usable as names in some positions
        if tok.kind in (TokenKind.TRUE, TokenKind.FALSE, TokenKind.NULL):
            raise SyntaxError_(
                "expected name",
                locations=[SourceLocation(tok.line, tok.column)],
            )
        if tok.kind is TokenKind.NAME or (
            tok.kind is TokenKind.COMPOUND_NAME
        ):
            self.lex.advance()
            return str(tok.value)
        # Allow SQL keywords as identifiers when used as names
        if tok.kind is TokenKind.NAME:
            pass
        if isinstance(tok.value, str) and tok.kind is TokenKind.NAME:
            self.lex.advance()
            return tok.value
        # Many clause keywords are lexed as NAME already. If we see a NAME-like
        # reserved word used as identifier — lexer returns NAME for non-bool.
        raise SyntaxError_(
            "expected name",
            locations=[SourceLocation(tok.line, tok.column)],
        )

    def _expect_kw(self, word: str) -> Token:
        tok = self.lex.peek()
        if not self._is_kw(tok, word):
            raise SyntaxError_(
                f"expected {word}",
                locations=[SourceLocation(tok.line, tok.column)],
            )
        return self.lex.advance()

    @staticmethod
    def _is_kw(tok: Token, word: str) -> bool:
        if tok.kind is TokenKind.NAME and isinstance(tok.value, str):
            return tok.value.upper() == word.upper()
        if word.upper() == "TRUE" and tok.kind is TokenKind.TRUE:
            return True
        if word.upper() == "FALSE" and tok.kind is TokenKind.FALSE:
            return True
        if word.upper() == "NULL" and tok.kind is TokenKind.NULL:
            return True
        return False


def parse(source: str, *, flags: FeatureFlags | None = None) -> A.SelectStmt:
    return Parser(source, flags=flags).parse_query()


def parse_expr(source: str, *, flags: FeatureFlags | None = None) -> A.Expr:
    return Parser(source, flags=flags).parse_expr_only()
