"""Mbarrier (memory barrier) state machine. Per-CTA pool keyed by smem byte offset.
Phase semantics: barrier flips between phase 0 and phase 1 as arrived_count reaches
expected_count. try_wait(phase=p) returns True iff the barrier has *flipped past*
phase p (i.e., bar.phase != p)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Mbarrier:
    expected_count: int
    arrived_count: int = 0
    phase: int = 0
    pending_tx: list[tuple[int, int]] = field(default_factory=list)
    """Each tuple = (tx_bytes, completion_at). When SM ticks past completion_at,
    one arrive (with tx_bytes weight) is registered."""


class MbarrierPool:
    """Per-CTA pool. SM holds one MbarrierPool per active CTA."""

    def __init__(self) -> None:
        self._barriers: dict[int, Mbarrier] = {}

    def init(self, smem_addr: int, expected: int) -> None:
        self._barriers[smem_addr] = Mbarrier(expected_count=expected)

    def arrive(self, smem_addr: int) -> None:
        bar = self._barriers[smem_addr]
        bar.arrived_count += 1
        if bar.arrived_count >= bar.expected_count:
            bar.arrived_count = 0
            bar.phase ^= 1

    def arrive_tx(self, smem_addr: int, tx_bytes: int, completion_at: int) -> None:
        bar = self._barriers[smem_addr]
        bar.pending_tx.append((tx_bytes, completion_at))

    def tick(self, now: int) -> None:
        """Drain pending_tx whose completion_at <= now. Each drain = 1 arrive."""
        for bar in self._barriers.values():
            new_pending: list[tuple[int, int]] = []
            for tx_bytes, comp in bar.pending_tx:
                if comp <= now:
                    bar.arrived_count += 1
                    if bar.arrived_count >= bar.expected_count:
                        bar.arrived_count = 0
                        bar.phase ^= 1
                else:
                    new_pending.append((tx_bytes, comp))
            bar.pending_tx = new_pending

    def try_wait(self, smem_addr: int, expected_phase: int) -> bool:
        """True iff barrier has flipped past expected_phase (bar.phase != expected_phase)."""
        bar = self._barriers.get(smem_addr)
        if bar is None:
            return False
        return bar.phase != expected_phase
