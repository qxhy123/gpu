from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from gpusim.config.schema import CacheConfig
from gpusim.core.atomic import L2AtomicQueue
from .line import CacheSet
from .l2_mshr import L2Mshr


class HBMProtocol(Protocol):
    def request(self, line_addr: int, now: int) -> int: ...
    def write_request(self, line_addr: int, now: int) -> int: ...


@dataclass
class L2Line:
    """Logical L2 cache line metadata used for window/partitioning tracking."""
    addr: int
    valid: bool = False
    dirty: bool = False
    last_use: int = 0
    owner_stream_id: int = -1     # NEW Phase 8
    in_window: bool = False       # NEW Phase 8


class L2Cache:
    """Tag-precise L2 cache with write-back + write-allocate semantics +
    MSHR for cross-SM coalescing (Phase 4)."""

    def __init__(self, cfg: CacheConfig, hbm: HBMProtocol, recorder=None):
        self.cfg = cfg
        self._hbm = hbm
        self._recorder = recorder
        self._line_bytes = cfg.l2_line_bytes
        self._n_lines = cfg.l2_size_bytes // self._line_bytes
        self._n_sets = self._n_lines // cfg.l2_ways
        self._set_mask = self._n_sets - 1
        self._set_bits = (self._n_sets - 1).bit_length()
        self._sets: dict[int, CacheSet] = {
            i: CacheSet(ways=cfg.l2_ways) for i in range(self._n_sets)
        }
        self._mshr = L2Mshr(n_slots=cfg.l2_mshr_slots)
        self._atomic_queue = L2AtomicQueue(n_slots=cfg.atomic_queue_capacity)
        self._stream_windows: dict = {}    # NEW Phase 8 — stream_id -> (start_set, n_sets)

    def fetch(self, *, line_addr: int, now: int, sm_id: int = -1,
              requesting_stream_id: int = -1) -> int:
        set_idx = line_addr & self._set_mask
        tag = line_addr >> self._set_bits
        line = self._sets[set_idx].find(tag)

        if line is not None:                            # HIT
            self._sets[set_idx].touch(line)
            way = self._sets[set_idx]._lines.index(line)
            if self._recorder is not None:
                self._recorder.l2_access(
                    cycle=now, kind="HIT",
                    line_addr=line_addr, set_idx=set_idx, way=way,
                    origin_sm=line.origin_sm, hit_sm=sm_id,
                )
            return now + self.cfg.l2_hit_latency

        # MISS — try MSHR
        allocated, entry = self._mshr.lookup_or_alloc(
            line_addr=line_addr, sm_id=sm_id, now=now,
        )
        if entry is None:
            if self._recorder is not None:
                self._recorder.l2_mshr(
                    kind="FULL", cycle=now, line_addr=line_addr,
                    sm_id=sm_id, n_waiters=0,
                )
            return -1
        if not allocated:
            # MERGE
            if self._recorder is not None:
                self._recorder.l2_mshr(
                    kind="MERGE", cycle=now, line_addr=line_addr,
                    sm_id=sm_id, n_waiters=len(entry.waiters),
                )
            return max(entry.completion_at, now + self.cfg.l2_hit_latency)

        # New miss: fetch from HBM
        hbm_complete = self._hbm.request(line_addr, now)
        completion = hbm_complete + self.cfg.l2_miss_install_latency
        entry.completion_at = completion
        evicted = self._sets[set_idx].install(
            tag=tag, dirty=False, origin_sm=sm_id,
            requesting_stream_id=requesting_stream_id,
            line_in_window_check=self._line_in_window,
        )
        way = next(i for i, ln in enumerate(self._sets[set_idx]._lines)
                    if ln.tag == tag and ln.valid)
        if evicted is not None and evicted.dirty:
            evicted_addr = (evicted.tag << self._set_bits) | set_idx
            self._hbm.write_request(evicted_addr, hbm_complete)
            if self._recorder is not None:
                self._recorder.l2_access(
                    cycle=now, kind="EVICT_DIRTY",
                    line_addr=line_addr, set_idx=set_idx, way=way,
                    victim_addr=evicted_addr,
                    origin_sm=sm_id, hit_sm=sm_id,
                )
        else:
            if self._recorder is not None:
                kind = "EVICT_CLEAN" if evicted is not None else "MISS_LOAD"
                self._recorder.l2_access(
                    cycle=now, kind=kind,
                    line_addr=line_addr, set_idx=set_idx, way=way,
                    origin_sm=sm_id, hit_sm=sm_id,
                )
        if self._recorder is not None:
            self._recorder.l2_mshr(
                kind="ALLOC", cycle=now, line_addr=line_addr,
                sm_id=sm_id, n_waiters=1,
            )
        return completion

    def tick(self, now: int) -> None:
        ready = [e for e in self._mshr.in_flight()
                  if e.completion_at >= 0 and e.completion_at <= now]
        for entry in ready:
            self._mshr.release(entry.line_addr)
            if self._recorder is not None:
                self._recorder.l2_mshr(
                    kind="RELEASE", cycle=now, line_addr=entry.line_addr,
                    sm_id=entry.origin_sm, n_waiters=len(entry.waiters),
                )

    def write_through(self, line_addr: int, now: int, sm_id: int = -1,
                      requesting_stream_id: int = -1) -> None:
        set_idx = line_addr & self._set_mask
        tag = line_addr >> self._set_bits
        line = self._sets[set_idx].find(tag)
        if line is not None:
            self._sets[set_idx].touch(line)
            way = self._sets[set_idx]._lines.index(line)
            line.dirty = True
            if self._recorder is not None:
                self._recorder.l2_access(
                    cycle=now, kind="HIT",
                    line_addr=line_addr, set_idx=set_idx, way=way,
                    origin_sm=line.origin_sm, hit_sm=sm_id,
                )
            return
        self._hbm.request(line_addr, now)
        evicted = self._sets[set_idx].install(
            tag=tag, dirty=True, origin_sm=sm_id,
            requesting_stream_id=requesting_stream_id,
            line_in_window_check=self._line_in_window,
        )
        way = next(i for i, ln in enumerate(self._sets[set_idx]._lines)
                    if ln.tag == tag and ln.valid)
        if evicted is not None and evicted.dirty:
            evicted_addr = (evicted.tag << self._set_bits) | set_idx
            self._hbm.write_request(evicted_addr, now)
            if self._recorder is not None:
                self._recorder.l2_access(
                    cycle=now, kind="EVICT_DIRTY",
                    line_addr=line_addr, set_idx=set_idx, way=way,
                    victim_addr=evicted_addr,
                    origin_sm=sm_id, hit_sm=sm_id,
                )
        else:
            if self._recorder is not None:
                kind = "EVICT_CLEAN" if evicted is not None else "MISS_STORE"
                self._recorder.l2_access(
                    cycle=now, kind=kind,
                    line_addr=line_addr, set_idx=set_idx, way=way,
                    origin_sm=sm_id, hit_sm=sm_id,
                )

    def atomic_op(self, *, line_addr: int, sm_id: int, op: str,
                  op_kind: str, now: int) -> int:
        """Enqueue an atomic on this line; return completion cycle.
        Phase 6 simplification: assume atomic always L2-hit (line resident).
        """
        completion = self._atomic_queue.enqueue(
            line_addr=line_addr, sm_id=sm_id, op=op, op_kind=op_kind,
            arrival=now, atomic_op_latency=self.cfg.atomic_op_latency,
            l2_hit_latency=self.cfg.l2_hit_latency,
        )
        return completion

    # ------------------------------------------------------------------ #
    # Phase 8 — stream window partitioning                                #
    # ------------------------------------------------------------------ #

    def register_stream_window(self, stream_id: int, start_set: int, n_sets: int) -> None:
        """Reserve [start_set, start_set+n_sets) as a protected window for this stream."""
        self._stream_windows[stream_id] = (start_set, n_sets)

    def _line_in_window(self, line: "L2Line", set_idx: int) -> bool:
        """Return True if *line* resides in its owner's registered window."""
        if line.owner_stream_id < 0:
            return False
        window = self._stream_windows.get(line.owner_stream_id)
        if window is None:
            return False
        start, n = window
        return start <= set_idx < start + n
