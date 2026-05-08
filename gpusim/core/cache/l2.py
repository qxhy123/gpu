from __future__ import annotations
from typing import Protocol
from gpusim.config.schema import CacheConfig
from .line import CacheSet, CacheLine


class HBMProtocol(Protocol):
    def request(self, line_addr: int, now: int) -> int: ...
    def write_request(self, line_addr: int, now: int) -> int: ...


class L2Cache:
    """Tag-precise L2 cache with write-back + write-allocate semantics."""

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

    def fetch(self, *, line_addr: int, now: int) -> int:
        """L1 calls this on miss. Returns the cycle when L2 has the data ready
        for L1 to install."""
        set_idx = line_addr & self._set_mask
        tag = line_addr >> self._set_bits
        line = self._sets[set_idx].find(tag)

        if line is not None:                            # HIT
            self._sets[set_idx].touch(line)
            way = self._sets[set_idx]._lines.index(line)
            if self._recorder is not None:
                self._recorder.l2_access(cycle=now, kind="HIT",
                                         line_addr=line_addr, set_idx=set_idx, way=way)
            return now + self.cfg.l2_hit_latency

        # MISS — fetch from HBM
        hbm_complete = self._hbm.request(line_addr, now)
        # install (with potential dirty eviction)
        evicted = self._sets[set_idx].install(tag=tag, dirty=False)
        way = next(i for i, ln in enumerate(self._sets[set_idx]._lines)
                   if ln.tag == tag and ln.valid)
        if evicted is not None and evicted.dirty:
            evicted_addr = (evicted.tag << self._set_bits) | set_idx
            self._hbm.write_request(evicted_addr, hbm_complete)
            if self._recorder is not None:
                self._recorder.l2_access(cycle=now, kind="EVICT_DIRTY",
                                         line_addr=line_addr, set_idx=set_idx, way=way,
                                         victim_addr=evicted_addr)
        else:
            if self._recorder is not None:
                kind = "EVICT_CLEAN" if evicted is not None else "MISS_LOAD"
                self._recorder.l2_access(cycle=now, kind=kind,
                                         line_addr=line_addr, set_idx=set_idx, way=way)
        return hbm_complete + self.cfg.l2_miss_install_latency

    def write_through(self, line_addr: int, now: int) -> None:
        """L1 calls this on store-miss (no-write-allocate at L1) or store-hit
        (write-through). Phase 2: write-allocate at L2 — fetch line if not present,
        mark it dirty."""
        set_idx = line_addr & self._set_mask
        tag = line_addr >> self._set_bits
        line = self._sets[set_idx].find(tag)
        if line is not None:                            # HIT — just mark dirty
            self._sets[set_idx].touch(line)
            way = self._sets[set_idx]._lines.index(line)
            line.dirty = True
            if self._recorder is not None:
                self._recorder.l2_access(cycle=now, kind="HIT",
                                         line_addr=line_addr, set_idx=set_idx, way=way)
            return
        # MISS — write-allocate (fetch line from HBM, mark dirty)
        self._hbm.request(line_addr, now)
        evicted = self._sets[set_idx].install(tag=tag, dirty=True)
        way = next(i for i, ln in enumerate(self._sets[set_idx]._lines)
                   if ln.tag == tag and ln.valid)
        if evicted is not None and evicted.dirty:
            evicted_addr = (evicted.tag << self._set_bits) | set_idx
            self._hbm.write_request(evicted_addr, now)
            if self._recorder is not None:
                self._recorder.l2_access(cycle=now, kind="EVICT_DIRTY",
                                         line_addr=line_addr, set_idx=set_idx, way=way,
                                         victim_addr=evicted_addr)
        else:
            if self._recorder is not None:
                kind = "EVICT_CLEAN" if evicted is not None else "MISS_STORE"
                self._recorder.l2_access(cycle=now, kind=kind,
                                         line_addr=line_addr, set_idx=set_idx, way=way)
