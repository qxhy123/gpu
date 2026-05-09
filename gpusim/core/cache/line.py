from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CacheLine:
    tag: int
    valid: bool = False
    dirty: bool = False
    lru_pos: int = 0           # 0 = MRU; ways-1 = LRU
    origin_sm: int = -1
    owner_stream_id: int = -1  # NEW Phase 9 — stream that owns this line
    in_window: bool = False    # NEW Phase 9 — line is in owner's protected window


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

    def install(self, *, tag: int, dirty: bool,
                origin_sm: int = -1,
                requesting_stream_id: int = -1,
                line_in_window_check=None) -> "CacheLine | None":
        """Install a new line with this tag. Returns the evicted line (or None
        if a free way was used or all ways are window-protected).

        Phase 9 kwargs
        --------------
        requesting_stream_id : int
            Stream performing the install.  Lines owned by a *different* stream
            that pass ``line_in_window_check`` are excluded from eviction
            candidates.
        line_in_window_check : callable(CacheLine, set_idx) -> bool | None
            If provided, called for each valid line to decide whether it is
            window-protected.  Pass ``None`` to disable filtering (original
            behaviour).
        """
        # try invalid way first — free ways are always usable
        for line in self._lines:
            if not line.valid:
                # promote it to MRU
                old_pos = line.lru_pos
                line.tag = tag
                line.valid = True
                line.dirty = dirty
                line.origin_sm = origin_sm
                line.owner_stream_id = requesting_stream_id
                line.in_window = False
                line.lru_pos = 0
                # bump older lines down (those that were below old_pos)
                for other in self._lines:
                    if other is line:
                        continue
                    if other.valid and other.lru_pos < old_pos:
                        other.lru_pos += 1
                return None

        # all ways valid → build candidate list, honouring window protection
        set_idx = getattr(self, 'set_idx', 0)
        candidates = []
        for line in self._lines:
            if (line_in_window_check is not None
                    and line.owner_stream_id != requesting_stream_id
                    and line_in_window_check(line, set_idx)):
                # protected by another stream's window — skip
                continue
            candidates.append(line)

        if not candidates:
            # every way is window-protected by a different stream
            return None

        # evict LRU among candidates
        victim = max(candidates, key=lambda c: c.lru_pos)
        evicted = CacheLine(tag=victim.tag, valid=True,
                            dirty=victim.dirty, lru_pos=victim.lru_pos)
        # replace in-place
        victim.tag = tag
        victim.dirty = dirty
        victim.origin_sm = origin_sm
        victim.owner_stream_id = requesting_stream_id
        victim.in_window = False
        victim.lru_pos = 0
        # bump others
        for other in self._lines:
            if other is victim:
                continue
            other.lru_pos += 1
        return evicted
