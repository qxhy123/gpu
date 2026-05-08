from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional


class PtxType(Enum):
    s32 = "s32"
    u32 = "u32"
    s64 = "s64"
    u64 = "u64"
    b32 = "b32"
    b64 = "b64"
    f32 = "f32"
    pred = "pred"
    # Phase 3 additions
    f16  = "f16"
    bf16 = "bf16"
    e4m3 = "e4m3"
    e5m2 = "e5m2"
    tf32 = "tf32"
    s8   = "s8"
    u8   = "u8"
    s16  = "s16"


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
class RegGroup:
    """A `{reg0, reg1, ...}` operand group (e.g., mma matrix fragment)."""
    regs: tuple["Reg", ...]


@dataclass(frozen=True)
class TensorDescriptor:
    """Hopper TMA 2D descriptor (simplified — no swizzle, no multicast).
    IR-level static metadata extracted from `gpusim.tma_desc` instr."""
    gmem_base_reg: str
    dim_x: int
    dim_y: int
    stride_y: int
    elem_bytes: int


@dataclass(frozen=True)
class MbarrierHandle:
    """Pointer to mbarrier in shared memory (smem byte offset)."""
    smem_addr: int


@dataclass(frozen=True)
class Param:
    name: str
    type: PtxType


# operands are reg | imm | reg-group | param-name (resolved during parse)
Operand = Reg | Imm | RegGroup


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
    type: Optional[PtxType]   # None for ops with no type modifier (e.g. wgmma.fence)
    pc: int
    src_loc: SrcLoc


@dataclass(frozen=True)
class Kernel:
    name: str
    params: tuple[Param, ...]
    regs: RegDecl
    instrs: tuple[Instr, ...]
    labels: Mapping[str, int]
    ipdom: Mapping[int, int]
