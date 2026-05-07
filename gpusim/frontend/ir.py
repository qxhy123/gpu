from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PtxType(Enum):
    s32 = "s32"
    u32 = "u32"
    s64 = "s64"
    u64 = "u64"
    b32 = "b32"
    b64 = "b64"
    f32 = "f32"
    pred = "pred"


class MemSpace(Enum):
    GLOBAL = "global"
    SHARED = "shared"
    PARAM = "param"
    LOCAL = "local"


@dataclass(frozen=True)
class SrcLoc:
    file: str
    line: int


@dataclass(frozen=True)
class Reg:
    name: str
    type: PtxType


@dataclass(frozen=True)
class Imm:
    value: int | float
    type: PtxType


@dataclass(frozen=True)
class Param:
    name: str
    type: PtxType


# operands are reg | imm | param-name (resolved during parse)
Operand = Reg | Imm


@dataclass(frozen=True)
class Predicate:
    reg: str
    negated: bool


@dataclass(frozen=True)
class RegDecl:
    s32: int = 0
    u32: int = 0
    s64: int = 0
    u64: int = 0
    b32: int = 0
    b64: int = 0
    f32: int = 0
    pred: int = 0


@dataclass(frozen=True)
class Instr:
    op: str
    dst: tuple[Operand, ...]
    src: tuple[Operand | str, ...]   # str = label or param name
    pred: Optional[Predicate]
    space: Optional[MemSpace]
    type: PtxType
    pc: int
    src_loc: SrcLoc


@dataclass(frozen=True)
class Kernel:
    name: str
    params: tuple[Param, ...]
    regs: RegDecl
    instrs: tuple[Instr, ...]
    labels: dict[str, int]
    ipdom: dict[int, int]
