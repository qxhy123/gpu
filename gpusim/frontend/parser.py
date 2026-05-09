from __future__ import annotations
from typing import Iterator
from types import MappingProxyType
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
            labels=MappingProxyType(labels),
            ipdom=MappingProxyType({}),
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
        # RegDecl tracks the 8 base types; other phase-3 types (f16, bf16, etc.) are parsed but ignored
        _RECDECL_TYPES = frozenset(("s32","u32","s64","u64","b32","b64","f32","pred"))
        counts = {k: 0 for k in _RECDECL_TYPES}
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
            if ty.value in _RECDECL_TYPES:
                counts[ty.value] = count
            # else: phase-3 types like f16, bf16 not tracked in RegDecl
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

        # opcode dotted with optional :: segments: ident( '.' ident | '.' num ident? | '::' ident )*
        # e.g. cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        op_parts = [self.eat("IDENT").value]
        while True:
            t = self.peek()
            if t.kind == "DOT" and self.peek(1).kind == "IDENT":
                self.eat("DOT")
                op_parts.append("." + self.eat("IDENT").value)
            elif t.kind == "DOT" and self.peek(1).kind == "NUM":
                # handle segments like ".2d" which lex as DOT NUM IDENT
                self.eat("DOT")
                num_part = self.eat("NUM").value
                seg = "." + num_part
                if self.peek().kind == "IDENT":
                    seg += self.eat("IDENT").value
                op_parts.append(seg)
            elif t.kind == "COLONCOLON" and self.peek(1).kind == "IDENT":
                self.eat("COLONCOLON")
                op_parts.append("::" + self.eat("IDENT").value)
            else:
                break
        op = "".join(op_parts)

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
    def _type_from_op(op: str) -> PtxType | None:
        # last dotted component that is a known type (only dotted, ignore :: segments)
        for part in reversed(op.split(".")):
            try:
                return PtxType(part)
            except ValueError:
                continue
        return None  # mma/wgmma/cp.async/mbarrier/bra/bar.sync etc.

    def _parse_operands(self, op: str, ty: PtxType | None) -> tuple[list[Operand], list]:
        if op == "gpusim.tma_desc":
            # gpusim.tma_desc %handle, %gmem_base, dim_x, dim_y, stride_y, elem_bytes;
            handle = self._parse_operand(PtxType.u64)
            self.eat("COMMA")
            gmem_base = self._parse_operand(PtxType.u64)
            self.eat("COMMA")
            dim_x = self._parse_operand(PtxType.s32)
            self.eat("COMMA")
            dim_y = self._parse_operand(PtxType.s32)
            self.eat("COMMA")
            stride_y = self._parse_operand(PtxType.s32)
            self.eat("COMMA")
            elem_bytes = self._parse_operand(PtxType.s32)
            return [handle], [gmem_base, dim_x, dim_y, stride_y, elem_bytes]

        if op.startswith("mma.sync.") or op.startswith("wgmma.mma_async."):
            # dst-group, src-A, src-B[, src-C]
            dst_grp = self._parse_brace_list(PtxType.f32)
            self.eat("COMMA")
            src_a = self._parse_brace_list_or_reg(PtxType.f16)
            self.eat("COMMA")
            src_b = self._parse_brace_list_or_reg(PtxType.f16)
            srcs: list = [src_a, src_b]
            if self.accept("COMMA"):
                src_c = self._parse_brace_list_or_reg(PtxType.f32)
                srcs.append(src_c)
            return [dst_grp], srcs

        if op in ("wgmma.fence.sync.aligned", "wgmma.commit_group.sync.aligned"):
            return [], []
        if op == "wgmma.wait_group.sync.aligned":
            n_imm = self._parse_operand(PtxType.s32)
            return [], [n_imm]

        if op.startswith("cp.async.bulk.tensor."):
            # Load form has "mbarrier::complete_tx::bytes" in opcode (3 args).
            # Store form (Phase 4) ends in "global.shared::cta" (2 args).
            n_args = 3 if "mbarrier" in op else 2
            srcs: list = []
            for _ in range(n_args):
                self.eat("LBRACK")
                addr = self._parse_operand(PtxType.u64)
                self.eat("RBRACK")
                srcs.append(addr)
                if not self.accept("COMMA"):
                    break
            return [], srcs

        if op == "cp.async.bulk.commit_group":
            return [], []
        if op == "cp.async.bulk.wait_group":
            n_imm = self._parse_operand(PtxType.s32)
            return [], [n_imm]

        if op.startswith("mbarrier.init."):
            self.eat("LBRACK"); addr = self._parse_operand(PtxType.u64); self.eat("RBRACK")
            self.eat("COMMA"); count = self._parse_operand(PtxType.s32)
            return [], [addr, count]
        if op.startswith("mbarrier.arrive."):
            self.eat("LBRACK"); addr = self._parse_operand(PtxType.u64); self.eat("RBRACK")
            return [], [addr]
        if op.startswith("mbarrier.try_wait."):
            # %pred, [addr], phase
            pred_dst = self._parse_operand(PtxType.pred)
            self.eat("COMMA")
            self.eat("LBRACK"); addr = self._parse_operand(PtxType.u64); self.eat("RBRACK")
            self.eat("COMMA")
            phase = self._parse_operand(PtxType.s32)
            return [pred_dst], [addr, phase]

        if op == "ret":
            return [], []
        if op == "bra" or op.endswith(".bra"):
            # bra LABEL;
            label = self._parse_operand(ty or PtxType.b32)
            return [], [label]
        if op.startswith("bar."):
            # bar.sync N;  (N optional, default 0)
            srcs: list = []
            if self.peek().kind == "NUM":
                srcs.append(self._parse_operand(PtxType.s32))
            return [], srcs
        # Phase 5: barrier.cluster.{arrive,wait} — no operands
        if op in ("barrier.cluster.arrive", "barrier.cluster.wait"):
            return [], []
        if op.startswith("membar"):
            return [], []

        if op.startswith("atom.global.") or op.startswith("atom.shared."):
            # atom.<space>.<op>.<ty>  dst, [addr], val  (or val_cmp, val_swap for cas)
            is_cas = ".cas." in op
            ty = self._type_from_op(op) or PtxType.u32
            dst = self._parse_operand(ty)
            self.eat("COMMA")
            self.eat("LBRACK")
            addr = self._parse_operand(PtxType.u64)
            self.eat("RBRACK")
            self.eat("COMMA")
            atom_srcs: list = [addr, self._parse_operand(ty)]
            if is_cas:
                self.eat("COMMA")
                atom_srcs.append(self._parse_operand(ty))
            return [dst], atom_srcs

        if op.startswith("red.global.") or op.startswith("red.shared."):
            # red.<space>.<op>.<ty>  [addr], val  (no dst)
            ty = self._type_from_op(op) or PtxType.u32
            self.eat("LBRACK")
            addr = self._parse_operand(PtxType.u64)
            self.eat("RBRACK")
            self.eat("COMMA")
            val = self._parse_operand(ty)
            return [], [addr, val]

        if op.startswith("ld."):
            # ld.<space>.<ty> dst, [addr];
            dst = self._parse_operand(ty or PtxType.b32)
            self.eat("COMMA")
            base, off = self._parse_addr()
            ld_srcs: list = [base]
            if off is not None:
                ld_srcs.append(off)
            return [dst], ld_srcs

        if op.startswith("st."):
            # st.<space>.<ty> [addr], src;
            base, off = self._parse_addr()
            self.eat("COMMA")
            src = self._parse_operand(ty or PtxType.b32)
            st_srcs: list = [base]
            if off is not None:
                st_srcs.append(off)
            st_srcs.append(src)
            return [], st_srcs

        # arithmetic / mov / cvt / setp: dst, src...
        ops = list(self._parse_operand_list(ty or PtxType.b32))
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

    def _parse_brace_list(self, ty: PtxType = PtxType.b32) -> "RegGroup":
        from .ir import RegGroup
        self.eat("LBRACE")
        regs: list[Reg] = []
        while True:
            t = self.peek()
            if t.kind != "REG":
                raise ParseError(f"{t.file}:{t.line}:{t.col}: expected REG inside brace list, got {t.kind}")
            self.i += 1
            regs.append(Reg(name=t.value, type=ty))
            if self.accept("COMMA"):
                continue
            self.eat("RBRACE")
            break
        return RegGroup(regs=tuple(regs))

    def _parse_brace_list_or_reg(self, ty: PtxType) -> "Operand":
        if self.peek().kind == "LBRACE":
            return self._parse_brace_list(ty)
        return self._parse_operand(ty)

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
        labels=MappingProxyType(dict(k.labels)),
        ipdom=MappingProxyType(compute_ipdom(k)),
    )
