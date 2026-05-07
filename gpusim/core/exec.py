from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import struct
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


class GlobalMemory:
    """Flat byte-addressable global memory. Each `bind`-ed numpy array gets a base
    address and shares storage with the underlying buffer."""

    def __init__(self):
        self._next_base = 0x1_0000_0000  # arbitrary base above 32-bit imm range
        self._segments: dict[int, np.ndarray] = {}   # base -> buffer (1d uint8 view)
        self._names: dict[str, int] = {}

    def bind(self, name: str, arr: np.ndarray) -> int:
        view = arr.view(np.uint8).reshape(-1)
        base = self._next_base
        self._segments[base] = view
        self._names[name] = base
        # advance, aligned to 256 bytes
        self._next_base += (view.nbytes + 255) & ~255
        return base

    def address_of(self, name: str) -> int:
        return self._names[name]

    def _seg_for(self, addr: int) -> tuple[np.ndarray, int]:
        # find segment whose base <= addr < base + size
        for base, buf in self._segments.items():
            if base <= addr < base + buf.nbytes:
                return buf, addr - base
        raise ValueError(f"unbound global address 0x{addr:x}")

    def load_f32(self, addr: int) -> float:
        buf, off = self._seg_for(addr)
        return float(buf[off:off+4].view(np.float32)[0])

    def store_f32(self, addr: int, v: float) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+4].view(np.float32)[0] = np.float32(v)

    def load_u32(self, addr: int) -> int:
        buf, off = self._seg_for(addr)
        return int(buf[off:off+4].view(np.uint32)[0])

    def store_u32(self, addr: int, v: int) -> None:
        buf, off = self._seg_for(addr)
        buf[off:off+4].view(np.uint32)[0] = np.uint32(v & 0xFFFFFFFF)


class SharedMemory:
    def __init__(self, size_bytes: int = 48 * 1024):
        self.size_bytes = size_bytes
        self._cta: dict[int, np.ndarray] = {}

    def allocate_cta(self, cta_id: int, size_bytes: int) -> None:
        self._cta[cta_id] = np.zeros(size_bytes, dtype=np.uint8)

    def free_cta(self, cta_id: int) -> None:
        self._cta.pop(cta_id, None)

    def load_f32(self, cta_id: int, offset: int) -> float:
        return float(self._cta[cta_id][offset:offset+4].view(np.float32)[0])

    def store_f32(self, cta_id: int, offset: int, value: float) -> None:
        self._cta[cta_id][offset:offset+4].view(np.float32)[0] = np.float32(value)

    def load_u32(self, cta_id: int, offset: int) -> int:
        return int(self._cta[cta_id][offset:offset+4].view(np.uint32)[0])

    def store_u32(self, cta_id: int, offset: int, value: int) -> None:
        self._cta[cta_id][offset:offset+4].view(np.uint32)[0] = np.uint32(value & 0xFFFFFFFF)


class ParamSpace:
    def __init__(self, params: dict[str, int]):
        self._params = dict(params)

    def read_u64(self, name: str) -> int:
        return int(self._params[name])

    def read_u32(self, name: str) -> int:
        return int(self._params[name]) & 0xFFFFFFFF
