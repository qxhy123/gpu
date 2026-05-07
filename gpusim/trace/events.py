from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EventKind(Enum):
    CTA_LAUNCH = "CTA_LAUNCH"
    CTA_RETIRE = "CTA_RETIRE"
    INSTR_ISSUE = "INSTR_ISSUE"
    SMEM_ACCESS = "SMEM_ACCESS"
    GMEM_ACCESS = "GMEM_ACCESS"
    DIV_PUSH = "DIV_PUSH"
    DIV_POP = "DIV_POP"
    BAR_REACH = "BAR_REACH"
    BAR_RELEASE = "BAR_RELEASE"


@dataclass(frozen=True)
class WarpStateSegment:
    warp_id: int
    start: int
    end: int       # inclusive
    state: str
    pc: int


@dataclass(frozen=True)
class InstrIssueEvent:
    cycle: int
    warp_id: int
    pc: int
    op: str
    src_loc: tuple[str, int]
    active_mask: int


@dataclass(frozen=True)
class SmemEvent:
    cycle: int
    warp_id: int
    conflict_degree: int
    addresses: tuple[int, ...]


@dataclass(frozen=True)
class GmemEvent:
    cycle: int
    warp_id: int
    n_transactions: int
    efficiency: float
    addresses: tuple[int, ...]


@dataclass(frozen=True)
class DivEvent:
    kind: str        # "PUSH" | "POP"
    cycle: int
    warp_id: int
    pc: int
    rpc: int = -1
    taken_mask: int = 0


@dataclass(frozen=True)
class CtaEvent:
    kind: str        # "LAUNCH" | "RETIRE"
    cycle: int
    cta_id: int
    warps: int = 0
    regs: int = 0
    smem_bytes: int = 0


@dataclass(frozen=True)
class BarEvent:
    kind: str        # "REACH" | "RELEASE"
    cycle: int
    cta_id: int
    barrier_id: int
