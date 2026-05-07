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
        instrs: list[Instr] = []
        labels: dict[str, int] = {}
        while self.peek().kind != "RBRACE":
            # label "L1:"
            if self.peek().kind == "IDENT" and self.peek(1).kind == "COLON":
                labels[self.peek().value] = len(instrs)
                self.i += 2
                continue
            instrs.append(self._parse_instr(len(instrs)))
        return instrs, labels

    def _parse_instr(self, pc: int) -> Instr:
        loc_tok = self.peek()
        # optional predicate "@p" or "@!p"
        pred: Predicate | None = None
        if self.accept("AT"):
            negated = self.accept("BANG") is not None
            pred_reg = self.eat("REG").value
            pred = Predicate(reg=pred_reg, negated=negated)

        # opcode dotted: ident('.' ident)*
        op_parts = [self.eat("IDENT").value]
        while self.peek().kind == "DOT" and self.peek(1).kind == "IDENT":
            self.eat("DOT")
            op_parts.append(self.eat("IDENT").value)
        op = ".".join(op_parts)

        space = self._space_from_op(op)
        ptx_type = self._type_from_op(op)

        dst, src = self._parse_operands(op, ptx_type)
        self.eat("SEMI")
        return Instr(
            op=op, dst=tuple(dst), src=tuple(src), pred=pred,
            space=space, type=ptx_type, pc=pc,
            src_loc=SrcLoc(loc_tok.file, loc_tok.line),
        )

    @staticmethod
    def _space_from_op(op: str) -> MemSpace | None:
        for s in MemSpace:
            if f".{s.value}." in f".{op}.":
                return s
        return None

    @staticmethod
    def _type_from_op(op: str) -> PtxType:
        # last dotted component that is a known type
        for part in reversed(op.split(".")):
            try:
                return PtxType(part)
            except ValueError:
                continue
        return PtxType.b32  # fallback for branches and bar.sync etc.

    def _parse_operands(self, op: str, ty: PtxType) -> tuple[list[Operand], list]:
        if op == "bra" or op.endswith(".bra"):
            # bra LABEL;
            label = self._parse_operand(ty)
            return [], [label]
        if op.startswith("bar."):
            # bar.sync N;  (N optional, default 0)
            srcs: list = []
            if self.peek().kind == "NUM":
                srcs.append(self._parse_operand(PtxType.s32))
            return [], srcs
        if op.startswith("membar"):
            return [], []

        if op.startswith("ld."):
            # ld.<space>.<ty> dst, [addr];
            dst = self._parse_operand(ty)
            self.eat("COMMA")
            base, off = self._parse_addr()
            srcs: list = [base]
            if off is not None:
                srcs.append(off)
            return [dst], srcs

        if op.startswith("st."):
            # st.<space>.<ty> [addr], src;
            base, off = self._parse_addr()
            self.eat("COMMA")
            src = self._parse_operand(ty)
            srcs_st: list = [base]
            if off is not None:
                srcs_st.append(off)
            srcs_st.append(src)
            return [], srcs_st

        # arithmetic / mov / cvt / setp: dst, src...
        ops = list(self._parse_operand_list(ty))
        if not ops:
            return [], []
        return [ops[0]], ops[1:]

    def _parse_operand_list(self, ty: PtxType):
        yield self._parse_operand(ty)
        while self.accept("COMMA"):
            yield self._parse_operand(ty)

    def _parse_operand(self, ty: PtxType) -> Operand:
        t = self.peek()
        if t.kind == "REG":
            self.i += 1
            return Reg(name=t.value, type=ty)
        if t.kind == "SREG":
            self.i += 1
            return Reg(name=t.value, type=PtxType.u32)
        if t.kind == "NUM":
            self.i += 1
            if t.value.startswith(("0x", "0X", "-0x", "-0X")):
                v: int | float = int(t.value, 16)
            elif "." in t.value or "e" in t.value.lower():
                v = float(t.value)
            else:
                v = int(t.value)
            return Imm(value=v, type=ty)
        if t.kind == "IDENT":
            # bare identifier: parameter name OR label (resolved later)
            self.i += 1
            return t.value  # plain str — caller treats as label/param
        raise ParseError(f"{t.file}:{t.line}:{t.col}: unexpected operand token {t.kind} {t.value!r}")

    def _parse_addr(self) -> tuple[Operand, Imm | None]:
        # [reg]  or  [reg+imm]  or  [reg-imm]  or  [param_name]
        self.eat("LBRACK")
        base = self._parse_operand(PtxType.u64)
        off: Imm | None = None
        nxt = self.peek()
        if nxt.kind == "NUM":  # signed-prefixed lexer already handles '-NN'
            off = Imm(value=int(nxt.value, 0), type=PtxType.s32)
            self.i += 1
        elif nxt.kind == "PLUS":
            self.i += 1  # consume '+'
            num_tok = self.eat("NUM")
            off = Imm(value=int(num_tok.value, 0), type=PtxType.s32)
        self.eat("RBRACK")
        return base, off


def parse(src: str, file: str = "<input>") -> Kernel:
    p = _Parser(src, file)
    k = p.parse_kernel()
    from .ipdom import compute_ipdom
    return Kernel(
        name=k.name, params=k.params, regs=k.regs, instrs=k.instrs,
        labels=k.labels, ipdom=compute_ipdom(k),
    )
