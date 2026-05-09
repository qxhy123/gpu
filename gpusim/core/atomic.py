from __future__ import annotations
from dataclasses import dataclass


@dataclass
class L2AtomicEntry:
    line_addr: int
    arrival_cycle: int
    completion_at: int
    sm_id: int
    op: str               # "add" | "min" | "max" | "exch" | "cas"
    op_kind: str          # "atom" | "red"


class L2AtomicQueue:
    """Per-line atomic FIFO. Multiple SMs hitting the same line serialize.

    Each atomic op takes atomic_op_latency cycles after the previous one finishes
    on that line. Different lines do not serialize against each other.
    """

    def __init__(self, n_slots: int = 32):
        self.n_slots = n_slots
        self._queues: dict[int, list[L2AtomicEntry]] = {}

    def enqueue(self, *, line_addr: int, sm_id: int, op: str, op_kind: str,
                  arrival: int, atomic_op_latency: int,
                  l2_hit_latency: int) -> int:
        """Queue an atomic; returns its completion_at cycle."""
        q = self._queues.setdefault(line_addr, [])
        # Drop entries already completed before arrival
        q = [e for e in q if e.completion_at > arrival]
        prev_done = q[-1].completion_at if q else 0
        start = max(arrival + l2_hit_latency, prev_done)
        completion = start + atomic_op_latency
        entry = L2AtomicEntry(
            line_addr=line_addr, arrival_cycle=arrival,
            completion_at=completion, sm_id=sm_id, op=op, op_kind=op_kind,
        )
        q.append(entry)
        self._queues[line_addr] = q
        return completion

    def queue_depth(self, line_addr: int, now: int) -> int:
        """Number of atomic ops on line still in-flight at cycle `now`."""
        q = self._queues.get(line_addr, [])
        return sum(1 for e in q if e.completion_at > now)
