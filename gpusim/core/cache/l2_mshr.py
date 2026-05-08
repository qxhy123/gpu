from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class L2MshrEntry:
    line_addr: int
    arrival_cycle: int
    origin_sm: int
    completion_at: int = -1
    waiters: list[tuple[int, int]] = field(default_factory=list)


class L2Mshr:
    """L2 MSHR pool. Coalesces concurrent miss requests for the same line
    coming from multiple SMs."""

    def __init__(self, n_slots: int = 32):
        self.n_slots = n_slots
        self._table: dict[int, L2MshrEntry] = {}

    def lookup_or_alloc(self, *, line_addr: int, sm_id: int,
                          now: int) -> tuple[bool, L2MshrEntry | None]:
        existing = self._table.get(line_addr)
        if existing is not None:
            existing.waiters.append((sm_id, len(existing.waiters)))
            return (False, existing)
        if len(self._table) >= self.n_slots:
            return (False, None)
        entry = L2MshrEntry(line_addr=line_addr, arrival_cycle=now,
                             origin_sm=sm_id)
        entry.waiters.append((sm_id, 0))
        self._table[line_addr] = entry
        return (True, entry)

    def release(self, line_addr: int) -> None:
        self._table.pop(line_addr, None)

    def active_count(self) -> int:
        return len(self._table)

    def in_flight(self):
        return list(self._table.values())
