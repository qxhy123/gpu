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
