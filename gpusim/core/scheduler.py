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
        if self._current is not None and candidates(self._current):
            return self._current
        for i in range(self.n):
            if candidates(i):
                self._current = i
                return i
        self._current = None
        return None
