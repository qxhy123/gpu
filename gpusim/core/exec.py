from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np


RegName = str


class ThreadState:
    """Per-lane scalar register state. Phase 1 keeps registers as Python ints/floats;
    we convert to/from numpy at memory-op boundaries."""

    def __init__(self):
        self._u32: dict[RegName, int] = {}
        self._s32: dict[RegName, int] = {}
        self._u64: dict[RegName, int] = {}
        self._f32: dict[RegName, float] = {}
        self._pred: dict[RegName, bool] = {}

    def set_u32(self, name: RegName, v: int) -> None: self._u32[name] = v & 0xFFFFFFFF
    def get_u32(self, name: RegName) -> int: return self._u32.get(name, 0)
    def set_s32(self, name: RegName, v: int) -> None:
        v &= 0xFFFFFFFF
        if v & 0x80000000: v -= 0x100000000
        self._s32[name] = v
    def get_s32(self, name: RegName) -> int: return self._s32.get(name, 0)
    def set_u64(self, name: RegName, v: int) -> None: self._u64[name] = v & ((1<<64)-1)
    def get_u64(self, name: RegName) -> int: return self._u64.get(name, 0)
    def set_f32(self, name: RegName, v: float) -> None: self._f32[name] = float(np.float32(v))
    def get_f32(self, name: RegName) -> float: return self._f32.get(name, 0.0)
    def set_pred(self, name: RegName, v: bool) -> None: self._pred[name] = bool(v)
    def get_pred(self, name: RegName) -> bool: return self._pred.get(name, False)


@dataclass
class WarpFnState:
    warp_size: int
    tids: tuple[int, ...]      # global thread IDs of the lanes (or per-CTA)
    pc: int = 0
    active_mask: int = 0
    threads: list[ThreadState] = field(default_factory=list)

    def __post_init__(self):
        if not self.threads:
            self.threads = [ThreadState() for _ in range(self.warp_size)]
        if self.active_mask == 0:
            self.active_mask = (1 << self.warp_size) - 1
