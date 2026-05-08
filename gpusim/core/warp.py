from __future__ import annotations
from enum import Enum
from dataclasses import dataclass, field
from gpusim.core.exec import WarpFnState
from gpusim.core.simt_stack import SIMTStack
from gpusim.core.scoreboard import Scoreboard


class StallReason(Enum):
    ISSUED = "ISSUED"
    IDLE = "IDLE"
    FETCH_EMPTY = "FETCH_EMPTY"
    SCOREBOARD = "SCOREBOARD"
    STRUCTURAL = "STRUCTURAL"
    OPERAND = "OPERAND"
    MEM_DEP = "MEM_DEP"
    BARRIER = "BARRIER"
    PRED_OFF = "PRED_OFF"
    DIVERGENCE_SERIAL = "DIVERGENCE_SERIAL"
    MSHR_FULL = "MSHR_FULL"


@dataclass
class Warp:
    warp_id: int
    kernel: object
    fn_state: WarpFnState | None = None
    stack: SIMTStack | None = None
    scoreboard: Scoreboard = field(default_factory=Scoreboard)
    barrier_pc: int = -1
    finished: bool = False
    cta_id: int = 0
    last_gmem: object | None = None
    outstanding_loads: list[int] = field(default_factory=list)
    last_operand_extra: int = 0
    executor: object | None = None  # per-warp executor override (for multi-CTA)
    _mshr_full_stall: bool = False
