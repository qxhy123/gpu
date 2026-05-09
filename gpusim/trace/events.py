from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class EventKind(Enum):
    CTA_LAUNCH = "CTA_LAUNCH"
    CTA_RETIRE = "CTA_RETIRE"
    INSTR_ISSUE = "INSTR_ISSUE"
    SMEM_ACCESS = "SMEM_ACCESS"
    GMEM_ACCESS = "GMEM_ACCESS"
    DIV_PUSH = "DIV_PUSH"
    DIV_POP = "DIV_POP"
    BAR_REACH = "BAR_REACH"
    BAR_RELEASE = "BAR_RELEASE"


@dataclass(frozen=True)
class WarpStateSegment:
    warp_id: int
    start: int
    end: int       # inclusive
    state: str
    pc: int


@dataclass(frozen=True)
class InstrIssueEvent:
    cycle: int
    warp_id: int
    pc: int
    op: str
    src_loc: tuple[str, int]
    active_mask: int
    stream_id: int = 0


@dataclass(frozen=True)
class SmemEvent:
    cycle: int
    warp_id: int
    conflict_degree: int
    addresses: tuple[int, ...]


@dataclass(frozen=True)
class GmemEvent:
    cycle: int
    warp_id: int
    n_transactions: int
    efficiency: float
    addresses: tuple[int, ...]
    stream_id: int = 0


@dataclass(frozen=True)
class DivEvent:
    kind: str        # "PUSH" | "POP"
    cycle: int
    warp_id: int
    pc: int
    rpc: int = -1
    taken_mask: int = 0


@dataclass(frozen=True)
class CtaEvent:
    kind: str        # "LAUNCH" | "RETIRE"
    cycle: int
    cta_id: int
    warps: int = 0
    regs: int = 0
    smem_bytes: int = 0


@dataclass(frozen=True)
class BarEvent:
    kind: str        # "REACH" | "RELEASE"
    cycle: int
    cta_id: int
    barrier_id: int
    stream_id: int = 0


@dataclass(frozen=True)
class L1Event:
    kind: str               # "HIT" | "MISS_NEW" | "MISS_MERGE"
    cycle: int
    warp_id: int
    line_addr: int
    set_idx: int
    way: int
    mshr_slot: int | None = None


@dataclass(frozen=True)
class L2Event:
    kind: str               # "HIT" | "MISS_LOAD" | "MISS_STORE" | "EVICT_CLEAN" | "EVICT_DIRTY"
    cycle: int
    line_addr: int
    set_idx: int
    way: int
    victim_addr: int = -1


@dataclass(frozen=True)
class HBMEvent:
    kind: str               # "READ" | "WRITE_BACK"
    row_kind: str           # "ROW_HIT" | "ROW_MISS"
    cycle: int
    served_at: int
    addr: int
    channel: int
    bank: int
    row: int
    queue_wait: int


@dataclass(frozen=True)
class MmaEvent:
    cycle: int
    warp_id: int
    pc: int
    precision: str
    shape_m: int
    shape_n: int
    shape_k: int
    accum_dtype: str
    flops_count: int
    stream_id: int = 0


@dataclass(frozen=True)
class WgmmaEvent:
    kind: str            # "ISSUE" | "COMMIT_GROUP" | "WAIT_GROUP" | "DRAIN"
    cycle: int
    warp_group_id: int
    pc: int
    precision: str = ""
    shape_m: int = 0
    shape_n: int = 0
    shape_k: int = 0
    accum_dtype: str = ""
    commit_group_id: int = -1
    wait_n: int = -1
    completion_at: int = -1


@dataclass(frozen=True)
class TmaEvent:
    cycle: int
    completion_at: int
    smem_dst: int
    gmem_base: int
    dim_x: int
    dim_y: int
    bytes_total: int
    n_cache_lines: int
    mbarrier_addr: int


@dataclass(frozen=True)
class MbarrierEvent:
    kind: str            # "INIT" | "ARRIVE" | "ARRIVE_TX" | "FLIP" | "TRY_WAIT"
    cycle: int
    cta_id: int
    smem_addr: int
    expected: int = 0
    arrived: int = 0
    phase: int = 0
    pred_result: bool = False


@dataclass(frozen=True)
class CtaDispatchEvent:
    cycle: int
    cta_id: int
    sm_id: int
    queue_position: int = 0
    active_warps_at_dispatch: int = 0
    stream_id: int = 0


@dataclass(frozen=True)
class L2MshrEvent:
    kind: str          # "ALLOC" | "MERGE" | "RELEASE" | "FULL"
    cycle: int
    line_addr: int
    sm_id: int
    n_waiters: int = 0
    stream_id: int = 0


@dataclass(frozen=True)
class BulkStoreEvent:
    kind: str          # "ISSUE" | "COMMIT_GROUP" | "WAIT_GROUP" | "DRAIN"
    cycle: int
    warp_group_id: int
    sm_id: int
    pc: int = 0
    smem_src: int = 0
    gmem_base: int = 0
    bytes_total: int = 0
    completion_at: int = -1
    commit_group_id: int = -1
    wait_n: int = -1
    stream_id: int = 0


@dataclass(frozen=True)
class BulkLoadEvent:
    kind: str          # "ISSUE" | "COMMIT_GROUP" | "WAIT_GROUP" | "DRAIN"
    cycle: int
    warp_group_id: int
    sm_id: int
    pc: int = 0
    smem_dst: int = 0
    gmem_base: int = 0
    bytes_total: int = 0
    completion_at: int = -1
    commit_group_id: int = -1
    wait_n: int = -1
    stream_id: int = 0


@dataclass(frozen=True)
class ClusterDispatchEvent:
    cycle: int
    cluster_id: int
    cluster_size: int
    sm_ids: tuple
    cta_ids: tuple
    queue_position: int = 0
    stream_id: int = 0


@dataclass(frozen=True)
class ClusterBarrierEvent:
    kind: str          # "ARRIVE" | "WAIT_BLOCK" | "WAIT_RELEASE"
    cycle: int
    cluster_id: int
    cta_id: int
    rank: int
    sm_id: int
    arrived_count: int = 0
    stream_id: int = 0


@dataclass(frozen=True)
class AtomicEvent:
    cycle: int
    sm_id: int
    warp_id: int
    kind: str            # "ATOM" | "RED"
    op: str              # "add" | "min" | "max" | "exch" | "cas"
    space: str           # "global" | "shared"
    line_addr: int
    latency: int
    n_lanes: int = 1
    queue_depth_before: int = 0
    stream_id: int = 0


@dataclass(frozen=True)
class KernelLaunch:
    stream_id: int
    kernel_name: str
    grid: tuple
    block: tuple
    launch_cycle: int
    complete_cycle: int
    n_ctas: int


# Canonical short aliases for Phase 7 stream_id API
InstrIssue = InstrIssueEvent
MemoryAccess = GmemEvent
BarrierEvent = BarEvent
ClusterDispatch = ClusterDispatchEvent
ClusterBarrier = ClusterBarrierEvent
CtaDispatch = CtaDispatchEvent
