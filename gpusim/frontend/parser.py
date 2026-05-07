from __future__ import annotations
from typing import Iterator
from .lexer import tokenize, Tok
from .ir import (
    Kernel, Param, RegDecl, Instr, Operand, Reg, Imm, Predicate,
    PtxType, MemSpace, SrcLoc,
)


class ParseError(Exception):
    pass


class _Parser:
    def __init__(self, src: str, file: str):
        self.tokens = [t for t in tokenize(src, file) if t.kind != "NL"]
        self.i = 0
        self.file = file

    # ---- low-level token utilities ----
    def peek(self, k: int = 0) -> Tok:
        return self.tokens[self.i + k]

    def eat(self, kind: str, value: str | None = None) -> Tok:
        t = self.peek()
        if t.kind != kind or (value is not None and t.value != value):
            raise ParseError(
                f"{t.file}:{t.line}:{t.col}: expected {kind}"
                + (f"={value!r}" if value else "")
                + f", got {t.kind}={t.value!r}"
            )
        self.i += 1
        return t

    def accept(self, kind: str, value: str | None = None) -> Tok | None:
        t = self.peek()
        if t.kind == kind and (value is None or t.value == value):
            self.i += 1
            return t
        return None

    # ---- top level ----
    def parse_kernel(self) -> Kernel:
        # optional .visible / .weak / etc. before .entry
        while self.accept("DOT"):
            t = self.eat("IDENT")
            if t.value == "entry":
                break
            # else: directive like 'visible', 'weak' — ignore
        name = self.eat("IDENT").value
        params = self._parse_params()
        self.eat("LBRACE")
        regs = self._parse_reg_decls()
        # instructions handled in later task — for now consume until RBRACE
        instrs, labels = self._parse_body(regs)
        self.eat("RBRACE")
        return Kernel(
            name=name,
            params=tuple(params),
            regs=regs,
            instrs=tuple(instrs),
            labels=labels,
            ipdom={},
        )

    def _parse_params(self) -> list[Param]:
        params: list[Param] = []
        self.eat("LPAREN")
        if self.accept("RPAREN"):
            return params
        while True:
            self.eat("DOT"); self.eat("IDENT", "param")
            self.eat("DOT"); ty_tok = self.eat("IDENT")
            try:
                ty = PtxType(ty_tok.value)
            except ValueError:
                raise ParseError(f"unknown param type {ty_tok.value!r}")
            name = self.eat("IDENT").value
            params.append(Param(name=name, type=ty))
            if self.accept("COMMA"):
                continue
            self.eat("RPAREN")
            break
        return params

    def _parse_reg_decls(self) -> RegDecl:
        counts = {k: 0 for k in ("s32","u32","s64","u64","b32","b64","f32","pred")}
        while True:
            # peek for ".reg"
            if not (self.peek().kind == "DOT" and self.peek(1).kind == "IDENT"
                    and self.peek(1).value == "reg"):
                break
            self.eat("DOT"); self.eat("IDENT", "reg")
            self.eat("DOT"); ty_tok = self.eat("IDENT")
            try:
                ty = PtxType(ty_tok.value)
            except ValueError:
                raise ParseError(f"unknown reg type {ty_tok.value!r}")
            # %name<N>;  e.g. %r<4>;
            self.eat("REG")
            count = 1
            if self.accept("LT"):
                count = int(self.eat("NUM").value)
                self.eat("GT")
            self.eat("SEMI")
            counts[ty.value] = count
        return RegDecl(**counts)

    def _parse_body(self, regs: RegDecl) -> tuple[list[Instr], dict[str,int]]:
        # placeholder for Tasks 5–7; for now just skip everything until matching RBRACE
        instrs: list[Instr] = []
        labels: dict[str, int] = {}
        depth = 1
        while depth > 0:
            t = self.peek()
            if t.kind == "LBRACE":
                depth += 1; self.i += 1
            elif t.kind == "RBRACE":
                depth -= 1
                if depth == 0:
                    break
                self.i += 1
            else:
                self.i += 1
        return instrs, labels


def parse(src: str, file: str = "<input>") -> Kernel:
    p = _Parser(src, file)
    return p.parse_kernel()
