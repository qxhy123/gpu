from __future__ import annotations
from enum import Enum
from gpusim.config.schema import FUConfig


class FUKind(Enum):
    FP32 = "fp32"
    INT = "int"
    LSU = "lsu"
    BRU = "bru"
    SYNC = "sync"
    TC = "tc"


class FUSet:
    """Per-sub-core set of functional units. Tracks issue-busy state."""

    def __init__(self, fu_cfg: FUConfig):
        self.cfg = fu_cfg
        self._issue_free_at: dict[FUKind, int] = {k: 0 for k in FUKind}

    def classify(self, op: str) -> FUKind:
        if op.startswith("atom.") or op.startswith("red."):
            return FUKind.LSU
        if op.startswith("mma.sync.") or op.startswith("wgmma.mma_async."):
            return FUKind.TC
        if op.startswith("wgmma."):
            # wgmma.fence/commit_group/wait_group: TC stream-control, route to TC
            return FUKind.TC
        if op.startswith("cp.async.bulk.commit_group") or op.startswith("cp.async.bulk.wait_group"):
            return FUKind.LSU
        if op.startswith("cp.async.bulk."):
            return FUKind.LSU
        if op.startswith("mbarrier."):
            return FUKind.SYNC
        if op == "gpusim.tma_desc":
            return FUKind.INT
        if op.startswith("ld.") or op.startswith("st.") or op.startswith("mov."):
            return FUKind.LSU
        if op == "bra" or op.endswith(".bra"):
            return FUKind.BRU
        if "bra" in op and op.startswith("@"):  # handle "@%p1 bra L1" form
            return FUKind.BRU
        if op.startswith("bar.") or op.startswith("membar"):
            return FUKind.SYNC
        if op.startswith(("add.f", "sub.f", "mul.f", "mad.f", "fma.f")):
            return FUKind.FP32
        if op.startswith("cvt."):
            return FUKind.INT
        return FUKind.INT

    def is_free(self, kind: FUKind, now: int) -> bool:
        return self._issue_free_at[kind] <= now

    def reserve(self, kind: FUKind, now: int, occupancy_cycles: int) -> None:
        self._issue_free_at[kind] = max(self._issue_free_at[kind], now) + occupancy_cycles

    def result_latency(self, op: str) -> int:
        c = self.cfg
        if op.startswith("ld.global."): return c.gmem_latency
        if op.startswith("ld.shared."): return c.smem_latency
        if op.startswith("ld.param."):  return 1
        if op.startswith("mov."):       return 1
        if op.startswith("st."):        return 0
        if op.startswith(("mad.", "fma.")): return c.fma_latency
        if op.startswith(("add.f", "sub.f", "mul.f")): return c.fp32_latency
        if op.startswith("cvt."):       return c.int32_latency
        if op == "bra" or op.endswith(".bra"): return c.bru_latency
        if op.startswith("setp."):      return c.int32_latency
        if op.startswith("bar.") or op.startswith("membar"): return 1
        return c.int32_latency

    def issue_occupancy(self, op: str, smem_conflict_degree: int = 1,
                        gmem_transactions: int = 1) -> int:
        if op.startswith("ld.shared.") or op.startswith("st.shared."):
            return max(1, smem_conflict_degree)
        if op.startswith("ld.global.") or op.startswith("st.global."):
            return max(1, gmem_transactions)
        return 1
