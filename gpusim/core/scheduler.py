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


class MultiStreamScheduler:
    """RR scheduler over multiple streams; intra-stream FIFO over grids.

    Each cycle, scheduler iterates streams in RR order; first stream with a
    dispatchable CTA wins. After a grid's CTAs are all dispatched AND the
    grid is marked retired (via mark_grid_retired), scheduler can advance
    to the next grid in that stream's pending queue.
    """

    def __init__(self, streams: list, policy: str = "rr"):
        self.streams = list(streams)
        self.policy = policy
        self.cursor = 0
        self._cta_iters: dict[int, _CtaIter] = {}

    def _ensure_inflight(self, s) -> bool:
        """Move next pending grid into inflight if stream idle. Return True if has work."""
        if s.inflight is None and s.pending:
            s.inflight = s.pending.popleft()
            self._cta_iters[s.stream_id] = _CtaIter(s.inflight.grid)
        return s.inflight is not None

    def _next_cta_for_stream(self, s):
        """Return next (cta_idx, grid_launch) for stream, or None."""
        if not self._ensure_inflight(s):
            return None
        it = self._cta_iters.get(s.stream_id)
        if it is None: return None
        cta = it.next()
        return cta

    def _find_sm_with_capacity(self, cta, available_sms):
        """Pick first SM that has capacity. Simple: first non-None."""
        for sm in available_sms:
            if getattr(sm, "cap", 1) > 0:
                return sm
        return None

    def next_dispatch(self, available_sms):
        """Try each stream in RR order; first dispatchable wins."""
        for _ in range(len(self.streams)):
            s = self.streams[self.cursor]
            self.cursor = (self.cursor + 1) % len(self.streams)
            cta = self._next_cta_for_stream(s)
            if cta is None: continue
            sm = self._find_sm_with_capacity(cta, available_sms)
            if sm is not None:
                return (s, cta, sm)
        return None

    def mark_grid_retired(self, s) -> None:
        """Caller signals: stream's current inflight grid is fully retired.
        Scheduler can now advance to next pending grid."""
        s.inflight = None
        self._cta_iters.pop(s.stream_id, None)
