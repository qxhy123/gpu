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
    """Round-robin CTA→SM dispatch."""

    def __init__(self):
        self._next = 0

    def pick(self, sms, occ):
        n = len(sms)
        if n == 0:
            return None
        for _ in range(n):
            sm = sms[self._next]
            self._next = (self._next + 1) % n
            if sm.can_admit_cta(occ):
                return sm
        return None


class GreedyCtaScheduler:
    """Greedy load-balanced CTA→SM dispatch — picks SM with fewest active warps."""

    def pick(self, sms, occ):
        eligible = [sm for sm in sms if sm.can_admit_cta(occ)]
        if not eligible:
            return None
        return min(eligible, key=lambda sm: sm.active_warp_count())


def make_cta_scheduler(policy: str):
    if policy == "rr":
        return RRCtaScheduler()
    if policy == "greedy":
        return GreedyCtaScheduler()
    raise ValueError(f"unknown cta_policy {policy!r}")
