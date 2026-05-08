from __future__ import annotations
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Waiter:
    warp_id: int
    dst_regs: tuple[str, ...]


@dataclass
class MSHREntry:
    line_addr: int
    issued_at: int
    expected_complete: int
    waiters: list[Waiter] = field(default_factory=list)
    slot_id: int = -1     # set by pool

    def add_waiter(self, warp_id: int, dst_regs: tuple[str, ...]) -> None:
        self.waiters.append(Waiter(warp_id=warp_id, dst_regs=dst_regs))


class MSHRPool:
    """Per-L1 pool of N MSHRs. Allocate / merge / release."""

    def __init__(self, slots: int = 16):
        self.slots = slots
        self._entries: dict[int, MSHREntry] = {}     # slot_id -> entry
        self._next_slot = 0

    def is_full(self) -> bool:
        return len(self._entries) >= self.slots

    def find_for_line(self, line_addr: int) -> MSHREntry | None:
        for e in self._entries.values():
            if e.line_addr == line_addr:
                return e
        return None

    def allocate(self, *, line_addr: int, issued_at: int, expected: int,
                 warp_id: int, dst_regs: tuple[str, ...]) -> MSHREntry | None:
        if self.is_full():
            return None
        slot_id = self._next_slot
        self._next_slot += 1
        e = MSHREntry(
            line_addr=line_addr,
            issued_at=issued_at,
            expected_complete=expected,
            slot_id=slot_id,
            waiters=[Waiter(warp_id=warp_id, dst_regs=dst_regs)],
        )
        self._entries[slot_id] = e
        return e

    def release(self, entry: MSHREntry) -> None:
        self._entries.pop(entry.slot_id, None)

    def active_entries(self) -> Iterator[MSHREntry]:
        return iter(list(self._entries.values()))
