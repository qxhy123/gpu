from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from gpusim.config.schema import CacheConfig
from .line import CacheSet, CacheLine
from .mshr import MSHRPool, MSHREntry


class L2Protocol(Protocol):
    """Anything L1 can call as its downstream cache."""
    def fetch(self, line_addr: int, now: int) -> int:
        """Return cycle when the line is ready to install in L1."""
        ...


@dataclass
class Hit:
    ready_at: int

@dataclass
class MissNewMSHR:
    ready_at: int
    mshr_slot: int

@dataclass
class MissMergeMSHR:
    ready_at: int
    mshr_slot: int

@dataclass
class Reject:
    pass


AccessResult = Hit | MissNewMSHR | MissMergeMSHR | Reject


class L1Cache:
    def __init__(self, cfg: CacheConfig, l2: L2Protocol, recorder=None):
        self.cfg = cfg
        self.l2 = l2
        self._recorder = recorder
        self._line_bytes = cfg.l1_line_bytes
        self._n_lines = cfg.l1_size_bytes // cfg.l1_line_bytes
        self._n_sets = self._n_lines // cfg.l1_ways
        # round up to power-of-2 if needed (assume already)
        self._set_mask = self._n_sets - 1
        self._set_bits = (self._n_sets - 1).bit_length()
        self._sets: dict[int, CacheSet] = {
            i: CacheSet(ways=cfg.l1_ways) for i in range(self._n_sets)
        }
        self._mshr = MSHRPool(slots=cfg.mshr_slots)
        self._pending_installs: list[tuple[int, int]] = []  # (line_addr, install_at)

    def access(self, *, line_addr: int, warp_id: int, dst_regs: tuple[str, ...],
               mode: str, now: int) -> AccessResult:
        set_idx = line_addr & self._set_mask
        tag = line_addr >> self._set_bits
        line = self._sets[set_idx].find(tag)

        # HIT
        if line is not None:
            self._sets[set_idx].touch(line)
            way = self._sets[set_idx]._lines.index(line)
            if mode == "store":
                self.l2.write_through(line_addr=line_addr, now=now)
            if self._recorder is not None:
                self._recorder.l1_access(cycle=now, warp_id=warp_id, kind="HIT",
                                         line_addr=line_addr, set_idx=set_idx, way=way)
            return Hit(ready_at=now + (1 if mode == "store" else self.cfg.l1_hit_latency))

        # store-miss: write-through to L2, no-write-allocate at L1
        if mode == "store":
            self.l2.write_through(line_addr=line_addr, now=now)
            # no L1 event for store-miss bypass (line wasn't in L1)
            return Hit(ready_at=now + 1)

        # load miss — try MSHR merge
        existing = self._mshr.find_for_line(line_addr)
        if existing is not None:
            existing.add_waiter(warp_id=warp_id, dst_regs=dst_regs)
            if self._recorder is not None:
                self._recorder.l1_access(
                    cycle=now, warp_id=warp_id, kind="MISS_MERGE",
                    line_addr=line_addr, set_idx=set_idx, way=-1,
                    mshr_slot=existing.slot_id,
                )
            return MissMergeMSHR(ready_at=existing.expected_complete,
                                 mshr_slot=existing.slot_id)

        if self._mshr.is_full():
            return Reject()

        # allocate new MSHR + downstream fetch
        l2_complete = self.l2.fetch(line_addr=line_addr, now=now)
        expected_complete = l2_complete + self.cfg.l1_miss_check_latency
        mshr = self._mshr.allocate(
            line_addr=line_addr, issued_at=now, expected=expected_complete,
            warp_id=warp_id, dst_regs=dst_regs,
        )
        if self._recorder is not None:
            self._recorder.l1_access(
                cycle=now, warp_id=warp_id, kind="MISS_NEW",
                line_addr=line_addr, set_idx=set_idx, way=-1,
                mshr_slot=mshr.slot_id,
            )
        # schedule install
        self._pending_installs.append((line_addr, expected_complete))
        return MissNewMSHR(ready_at=expected_complete, mshr_slot=mshr.slot_id)

    def install_completed_lines(self, *, now: int) -> list[int]:
        """Install any MSHR entries whose expected_complete <= now. Returns list
        of installed line_addrs (for caller to release MSHR / wake waiters)."""
        installed = []
        remaining = []
        for line_addr, install_at in self._pending_installs:
            if install_at <= now:
                set_idx = line_addr & self._set_mask
                tag = line_addr >> self._set_bits
                self._sets[set_idx].install(tag=tag, dirty=False)
                # release MSHR
                mshr = self._mshr.find_for_line(line_addr)
                if mshr is not None:
                    self._mshr.release(mshr)
                installed.append(line_addr)
            else:
                remaining.append((line_addr, install_at))
        self._pending_installs = remaining
        return installed

    @property
    def mshr(self) -> MSHRPool:
        return self._mshr
