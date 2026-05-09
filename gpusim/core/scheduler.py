from __future__ import annotations
from typing import Callable


class LRRScheduler:
    """Loose Round Robin: cycles warp_id, picking the next ready one."""

    def __init__(self, warp_count: int):
        self.n = warp_count
        self._next = 0

    def pick(self, now: int, candidates: Callable[[int], bool]) -> int | None:
        for offset in range(self.n):
            i = (self._next + offset) % self.n
            if candidates(i):
                self._next = (i + 1) % self.n
                return i
        return None


class GTOScheduler:
    """Greedy-Then-Oldest: stay on current warp until it stalls; otherwise pick the oldest ready."""

    def __init__(self, warp_count: int):
        self.n = warp_count
        self._current: int | None = None

    def pick(self, now: int, candidates: Callable[[int], bool]) -> int | None:
        if self._current is not None and self._current < self.n and candidates(self._current):
            return self._current
        for i in range(self.n):
            if candidates(i):
                self._current = i
                return i
        self._current = None
        return None


class RRCtaScheduler:
    """Round-robin CTA→SM dispatch with peek/commit (Phase 5)."""

    def __init__(self):
        self._next = 0
        self._pending_advance: int | None = None

    def peek(self, sms, occ, k: int = 1):
        n = len(sms)
        if n == 0 or k <= 0:
            return None
        candidates = []
        try_next = self._next
        for _ in range(n):
            sm = sms[try_next]
            if sm.can_admit_cta(occ):
                candidates.append(sm)
                next_after = (try_next + 1) % n
                if len(candidates) == k:
                    self._pending_advance = next_after
                    return candidates
            try_next = (try_next + 1) % n
        self._pending_advance = None
        return None

    def commit(self, k: int = 1):
        if self._pending_advance is not None:
            self._next = self._pending_advance
            self._pending_advance = None

    def pick(self, sms, occ):
        result = self.peek(sms, occ, k=1)
        if result is None:
            return None
        self.commit(k=1)
        return result[0]


class GreedyCtaScheduler:
    """Greedy load-balanced CTA→SM dispatch with peek/commit (Phase 5)."""

    def peek(self, sms, occ, k: int = 1):
        eligible = sorted(
            [sm for sm in sms if sm.can_admit_cta(occ)],
            key=lambda sm: sm.active_warp_count())
        if len(eligible) >= k:
            return eligible[:k]
        return None

    def commit(self, k: int = 1):
        pass

    def pick(self, sms, occ):
        result = self.peek(sms, occ, k=1)
        if result is None:
            return None
        return result[0]


def make_cta_scheduler(policy: str):
    if policy == "rr":
        return RRCtaScheduler()
    if policy == "greedy":
        return GreedyCtaScheduler()
    raise ValueError(f"unknown cta_policy {policy!r}")


class _CtaIter:
    """Walks CTAs (x,y,z) for a grid in linear order."""
    def __init__(self, grid: tuple):
        self.grid = grid
        self.x = self.y = self.z = 0
        self._done = False

    def next(self):
        if self._done: return None
        cta = (self.x, self.y, self.z)
        self.x += 1
        if self.x >= self.grid[0]:
            self.x = 0; self.y += 1
            if self.y >= self.grid[1]:
                self.y = 0; self.z += 1
                if self.z >= self.grid[2]:
                    self._done = True
        return cta


class ConcurrentStreamScheduler:
    """Per-cycle weighted RR over multiple streams, with event-block awareness.

    Each cycle, scheduler iterates streams and dispatches up to weight CTAs
    per stream (default high=4, normal=2, low=1). Event-blocked streams skipped.
    """

    def __init__(self, streams: list, priority_weights: dict | None = None):
        self.streams = list(streams)
        self.cursor = 0
        self._cta_iters: dict = {}
        self._priority_weights = priority_weights or {"high": 4, "normal": 2, "low": 1}

    def stream_weight(self, s) -> int:
        return self._priority_weights.get(getattr(s, "priority", "normal"), 2)

    def is_event_blocked(self, s, current_cycle: int) -> bool:
        # Phase 8 M3 will activate event_waits; for now nothing blocks
        for ev in getattr(s, "event_waits", []):
            if not ev.is_signaled(current_cycle): return True
        return False

    def _ensure_inflight(self, s) -> bool:
        if s.inflight is None and s.pending:
            head = s.pending.popleft()
            # Phase 8 M3 will add _RecordMarker handling; for now treat all as GridLaunch
            s.inflight = head
            self._cta_iters[s.stream_id] = _CtaIter(head.grid)
            s.in_flight_ctas = head.grid[0] * head.grid[1] * head.grid[2]
        return s.inflight is not None

    def _pick_sm(self, available_sms, cta):
        for sm in available_sms:
            if getattr(sm, "cap", 1) > 0:
                return sm
        return None

    def step(self, available_sms, current_cycle: int) -> list:
        """Returns list of (stream, cta, sm) dispatches for this cycle."""
        decisions = []
        for s in self.streams:
            if s.is_idle() and s.in_flight_ctas == 0: continue
            if self.is_event_blocked(s, current_cycle): continue
            weight = self.stream_weight(s)
            for _ in range(weight):
                if not available_sms: break
                if not self._ensure_inflight(s): break
                cta = self._cta_iters[s.stream_id].next()
                if cta is None: break
                sm = self._pick_sm(available_sms, cta)
                if sm is None: break
                decisions.append((s, cta, sm))
        return decisions

    def mark_grid_retired(self, s) -> None:
        s.inflight = None
        self._cta_iters.pop(s.stream_id, None)


# Phase 7 -> Phase 8 alias for backward compat
MultiStreamScheduler = ConcurrentStreamScheduler
