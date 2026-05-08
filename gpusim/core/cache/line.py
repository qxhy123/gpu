from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CacheLine:
    tag: int
    valid: bool = False
    dirty: bool = False
    lru_pos: int = 0      # 0 = MRU; ways-1 = LRU


@dataclass
class CacheSet:
    """Per-set storage with LRU bookkeeping. Stores `n_ways` CacheLine slots."""
    n_ways: int = 4
    _lines: list[CacheLine] = field(default_factory=list)

    def __init__(self, ways: int = 4):
        self.n_ways = ways
        self._lines = [CacheLine(tag=0, valid=False, lru_pos=i)
                       for i in range(ways)]

    @property
    def ways(self) -> list[CacheLine]:
        return list(self._lines)

    @property
    def lines(self) -> list[CacheLine]:
        return list(self._lines)

    def find(self, tag: int) -> CacheLine | None:
        for line in self._lines:
            if line.valid and line.tag == tag:
                return line
        return None

    def touch(self, hit_line: CacheLine) -> None:
        """Move `hit_line` to MRU; shift others up to LRU."""
        old_pos = hit_line.lru_pos
        for line in self._lines:
            if line is hit_line:
                line.lru_pos = 0
            elif line.valid and line.lru_pos < old_pos:
                line.lru_pos += 1

    def install(self, *, tag: int, dirty: bool) -> CacheLine | None:
        """Install a new line with this tag. Returns the evicted line (or None
        if a free way was used)."""
        # try invalid way first
        for line in self._lines:
            if not line.valid:
                # promote it to MRU
                old_pos = line.lru_pos
                line.tag = tag
                line.valid = True
                line.dirty = dirty
                line.lru_pos = 0
                # bump older lines down (those that were below old_pos)
                for other in self._lines:
                    if other is line:
                        continue
                    if other.valid and other.lru_pos < old_pos:
                        other.lru_pos += 1
                return None
        # all ways valid → evict LRU
        victim_idx, victim = max(
            enumerate(self._lines), key=lambda iv: iv[1].lru_pos
        )
        evicted = CacheLine(tag=victim.tag, valid=True,
                            dirty=victim.dirty, lru_pos=victim.lru_pos)
        # replace in-place
        victim.tag = tag
        victim.dirty = dirty
        victim.lru_pos = 0
        # bump others
        for other in self._lines:
            if other is victim:
                continue
            other.lru_pos += 1
        return evicted
