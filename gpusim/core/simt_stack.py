from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SIMTEntry:
    pc: int
    active_mask: int
    rpc: int  # reconverge PC; -1 means none


class SIMTStack:
    def __init__(self, warp_size: int, entry_pc: int):
        self._warp_size = warp_size
        full = (1 << warp_size) - 1
        self._stack: list[SIMTEntry] = [SIMTEntry(pc=entry_pc, active_mask=full, rpc=-1)]

    def top(self) -> SIMTEntry: return self._stack[-1]

    def is_done(self) -> bool: return not self._stack

    def update_top_pc(self, pc: int) -> None: self._stack[-1].pc = pc

    def update_top_mask(self, mask: int) -> None: self._stack[-1].active_mask = mask

    def maybe_pop(self) -> bool:
        popped = False
        while len(self._stack) > 1 and self._stack[-1].pc == self._stack[-1].rpc:
            self._stack.pop()
            popped = True
        return popped

    def diverge(self, taken_pc: int, fallthrough_pc: int,
                taken_mask: int, rpc: int) -> bool:
        cur = self._stack[-1]
        if taken_mask == cur.active_mask:
            cur.pc = taken_pc
            return False
        if taken_mask == 0:
            cur.pc = fallthrough_pc
            return False
        ft_mask = cur.active_mask & ~taken_mask
        full_mask = cur.active_mask
        # Replace current entry with reconverge frame (full mask at rpc), then push
        # the two divergent paths above it so they each pop when they reach rpc.
        cur.pc = rpc
        cur.active_mask = full_mask
        cur.rpc = -1
        self._stack.append(SIMTEntry(pc=fallthrough_pc, active_mask=ft_mask, rpc=rpc))
        self._stack.append(SIMTEntry(pc=taken_pc, active_mask=taken_mask, rpc=rpc))
        return True

    def end_warp(self) -> None:
        self._stack.clear()
