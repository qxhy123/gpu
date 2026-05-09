from __future__ import annotations
from collections import defaultdict
from typing import Iterator
from .events import (
    WarpStateSegment, InstrIssueEvent, SmemEvent, GmemEvent,
    DivEvent, CtaEvent, BarEvent,
    L1Event, L2Event, HBMEvent,
    MmaEvent, WgmmaEvent, TmaEvent, MbarrierEvent,
)


class Recorder:
    def __init__(self):
        # warp_state RLE: per warp_id, list of [start, end, state, pc]
        self._ws_segs: dict[int, list[list]] = defaultdict(list)
        # cur_state[warp_id] tracks the open segment
        self._cur_state: dict[int, list] = {}   # [start, end, state, pc]
        self._instr_issues: list[InstrIssueEvent] = []
        self._smem: list[SmemEvent] = []
        self._gmem: list[GmemEvent] = []
        self._div: list[DivEvent] = []
        self._cta: list[CtaEvent] = []
        self._bar: list[BarEvent] = []
        self._l1: list[L1Event] = []
        self._l2: list[L2Event] = []
        self._hbm: list[HBMEvent] = []
        self.mma_events: list[MmaEvent] = []
        self.wgmma_events: list[WgmmaEvent] = []
        self.tma_events: list[TmaEvent] = []
        self.mbarrier_events: list[MbarrierEvent] = []
        self.cta_dispatch_events: list = []
        self.l2_mshr_events: list = []
        self.bulk_store_events: list = []
        self.cluster_dispatch_events: list = []
        self.cluster_barrier_events: list = []
        self.atomic_events: list = []
        self.kernel_launch_events: list = []

    def warp_state(self, *, cycle: int, warp_id: int, state: str, pc: int) -> None:
        cur = self._cur_state.get(warp_id)
        if cur and cur[2] == state and cur[3] == pc and cur[1] + 1 == cycle:
            cur[1] = cycle
            return
        if cur:
            self._ws_segs[warp_id].append(cur)
        self._cur_state[warp_id] = [cycle, cycle, state, pc]

    def flush(self) -> None:
        for wid, cur in self._cur_state.items():
            self._ws_segs[wid].append(cur)
        self._cur_state.clear()

    def warp_state_segments(self, warp_id: int) -> Iterator[WarpStateSegment]:
        # ensure flushed
        if warp_id in self._cur_state:
            self._ws_segs[warp_id].append(self._cur_state[warp_id])
            del self._cur_state[warp_id]
        for s in self._ws_segs[warp_id]:
            yield WarpStateSegment(warp_id=warp_id, start=s[0], end=s[1],
                                   state=s[2], pc=s[3])

    def all_warp_segments(self) -> Iterator[WarpStateSegment]:
        # flush any open
        for wid in list(self._cur_state):
            self._ws_segs[wid].append(self._cur_state[wid])
            del self._cur_state[wid]
        for wid, segs in self._ws_segs.items():
            for s in segs:
                yield WarpStateSegment(warp_id=wid, start=s[0], end=s[1],
                                       state=s[2], pc=s[3])

    def instr_issue(self, *, cycle, warp_id, pc, op, src_loc, active_mask,
                    stream_id: int = 0) -> None:
        self._instr_issues.append(InstrIssueEvent(
            cycle=cycle, warp_id=warp_id, pc=pc, op=op,
            src_loc=tuple(src_loc), active_mask=int(active_mask),
            stream_id=stream_id))

    def instr_issues(self) -> list[InstrIssueEvent]:
        return list(self._instr_issues)

    def smem_access(self, *, cycle, warp_id, conflict_degree, addresses) -> None:
        self._smem.append(SmemEvent(cycle, warp_id, conflict_degree, tuple(addresses)))

    def smem_accesses(self) -> list[SmemEvent]:
        return list(self._smem)

    def gmem_access(self, *, cycle, warp_id, n_transactions, efficiency, addresses,
                    stream_id: int = 0) -> None:
        self._gmem.append(GmemEvent(cycle, warp_id, n_transactions, float(efficiency),
                                    tuple(addresses), stream_id))

    def gmem_accesses(self) -> list[GmemEvent]:
        return list(self._gmem)

    def div_push(self, *, cycle, warp_id, pc, rpc, taken_mask) -> None:
        self._div.append(DivEvent("PUSH", cycle, warp_id, pc, rpc, taken_mask))

    def div_pop(self, *, cycle, warp_id, pc) -> None:
        self._div.append(DivEvent("POP", cycle, warp_id, pc, -1, 0))

    def div_events(self) -> list[DivEvent]:
        return list(self._div)

    def cta_launch(self, *, cycle, cta_id, warps, regs, smem_bytes) -> None:
        self._cta.append(CtaEvent("LAUNCH", cycle, cta_id, warps, regs, smem_bytes))

    def cta_retire(self, *, cycle, cta_id) -> None:
        self._cta.append(CtaEvent("RETIRE", cycle, cta_id))

    def cta_events(self) -> list[CtaEvent]:
        return list(self._cta)

    def bar_reach(self, *, cycle, cta_id, barrier_id=0, stream_id: int = 0) -> None:
        self._bar.append(BarEvent("REACH", cycle, cta_id, barrier_id, stream_id))

    def bar_release(self, *, cycle, cta_id, barrier_id=0, stream_id: int = 0) -> None:
        self._bar.append(BarEvent("RELEASE", cycle, cta_id, barrier_id, stream_id))

    def bar_events(self) -> list[BarEvent]:
        return list(self._bar)

    def l1_access(self, *, cycle, warp_id, kind, line_addr, set_idx, way,
                  mshr_slot=None) -> None:
        self._l1.append(L1Event(kind=kind, cycle=cycle, warp_id=warp_id,
                                line_addr=line_addr, set_idx=set_idx,
                                way=way, mshr_slot=mshr_slot))

    def l1_accesses(self) -> list[L1Event]:
        return list(self._l1)

    def l2_access(self, *, cycle, kind, line_addr, set_idx, way,
                  victim_addr: int = -1, origin_sm: int = -1,
                  hit_sm: int = -1) -> None:
        # origin_sm / hit_sm absorbed here; T27 will persist them in L2Event
        self._l2.append(L2Event(kind=kind, cycle=cycle, line_addr=line_addr,
                                set_idx=set_idx, way=way, victim_addr=victim_addr))

    def l2_mshr(self, *, kind: str, cycle: int, line_addr: int,
                  sm_id: int, n_waiters: int = 0,
                  stream_id: int = 0) -> None:
        from gpusim.trace.events import L2MshrEvent
        self.l2_mshr_events.append(L2MshrEvent(
            kind=kind, cycle=cycle, line_addr=line_addr,
            sm_id=sm_id, n_waiters=n_waiters,
            stream_id=stream_id,
        ))

    def l2_accesses(self) -> list[L2Event]:
        return list(self._l2)

    def hbm_access(self, *, cycle, served_at, addr, channel, bank, row,
                   kind, row_kind, queue_wait) -> None:
        self._hbm.append(HBMEvent(kind=kind, row_kind=row_kind, cycle=cycle,
                                  served_at=served_at, addr=addr, channel=channel,
                                  bank=bank, row=row, queue_wait=queue_wait))

    def hbm_accesses(self) -> list[HBMEvent]:
        return list(self._hbm)

    def mma(self, *, cycle: int, warp_id: int, pc: int, precision: str,
            shape_m: int, shape_n: int, shape_k: int, accum_dtype: str,
            flops_count: int, stream_id: int = 0) -> None:
        self.mma_events.append(MmaEvent(
            cycle=cycle, warp_id=warp_id, pc=pc, precision=precision,
            shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
            accum_dtype=accum_dtype, flops_count=flops_count,
            stream_id=stream_id,
        ))

    def wgmma(self, *, kind: str, cycle: int, warp_group_id: int, pc: int,
              precision: str = "", shape_m: int = 0, shape_n: int = 0,
              shape_k: int = 0, accum_dtype: str = "",
              commit_group_id: int = -1, wait_n: int = -1,
              completion_at: int = -1) -> None:
        self.wgmma_events.append(WgmmaEvent(
            kind=kind, cycle=cycle, warp_group_id=warp_group_id, pc=pc,
            precision=precision, shape_m=shape_m, shape_n=shape_n, shape_k=shape_k,
            accum_dtype=accum_dtype, commit_group_id=commit_group_id,
            wait_n=wait_n, completion_at=completion_at,
        ))

    def tma(self, *, cycle: int, completion_at: int, smem_dst: int,
            gmem_base: int, dim_x: int, dim_y: int, bytes_total: int,
            n_cache_lines: int, mbarrier_addr: int) -> None:
        self.tma_events.append(TmaEvent(
            cycle=cycle, completion_at=completion_at, smem_dst=smem_dst,
            gmem_base=gmem_base, dim_x=dim_x, dim_y=dim_y,
            bytes_total=bytes_total, n_cache_lines=n_cache_lines,
            mbarrier_addr=mbarrier_addr,
        ))

    def mbarrier(self, *, kind: str, cycle: int, cta_id: int, smem_addr: int,
                 expected: int = 0, arrived: int = 0, phase: int = 0,
                 pred_result: bool = False) -> None:
        self.mbarrier_events.append(MbarrierEvent(
            kind=kind, cycle=cycle, cta_id=cta_id, smem_addr=smem_addr,
            expected=expected, arrived=arrived, phase=phase, pred_result=pred_result,
        ))

    def cta_dispatch(self, *, cycle: int, cta_id: int, sm_id: int,
                       queue_position: int = 0,
                       active_warps_at_dispatch: int = 0,
                       stream_id: int = 0) -> None:
        from gpusim.trace.events import CtaDispatchEvent
        self.cta_dispatch_events.append(CtaDispatchEvent(
            cycle=cycle, cta_id=cta_id, sm_id=sm_id,
            queue_position=queue_position,
            active_warps_at_dispatch=active_warps_at_dispatch,
            stream_id=stream_id,
        ))

    def bulk_store(self, *, kind: str, cycle: int, warp_group_id: int,
                     sm_id: int, pc: int = 0,
                     smem_src: int = 0, gmem_base: int = 0,
                     bytes_total: int = 0, completion_at: int = -1,
                     commit_group_id: int = -1, wait_n: int = -1,
                     stream_id: int = 0) -> None:
        from gpusim.trace.events import BulkStoreEvent
        self.bulk_store_events.append(BulkStoreEvent(
            kind=kind, cycle=cycle, warp_group_id=warp_group_id, sm_id=sm_id,
            pc=pc, smem_src=smem_src, gmem_base=gmem_base,
            bytes_total=bytes_total, completion_at=completion_at,
            commit_group_id=commit_group_id, wait_n=wait_n,
            stream_id=stream_id,
        ))

    def cluster_dispatch(self, *, cycle: int, cluster_id: int,
                            cluster_size: int, sm_ids: tuple,
                            cta_ids: tuple, queue_position: int = 0,
                            stream_id: int = 0) -> None:
        from gpusim.trace.events import ClusterDispatchEvent
        self.cluster_dispatch_events.append(ClusterDispatchEvent(
            cycle=cycle, cluster_id=cluster_id, cluster_size=cluster_size,
            sm_ids=sm_ids, cta_ids=cta_ids, queue_position=queue_position,
            stream_id=stream_id,
        ))

    def cluster_barrier(self, *, kind: str, cycle: int, cluster_id: int,
                          cta_id: int, rank: int, sm_id: int,
                          arrived_count: int = 0,
                          stream_id: int = 0) -> None:
        from gpusim.trace.events import ClusterBarrierEvent
        self.cluster_barrier_events.append(ClusterBarrierEvent(
            kind=kind, cycle=cycle, cluster_id=cluster_id, cta_id=cta_id,
            rank=rank, sm_id=sm_id, arrived_count=arrived_count,
            stream_id=stream_id,
        ))

    def atomic(self, *, cycle: int, sm_id: int, warp_id: int,
                kind: str, op: str, space: str, line_addr: int,
                latency: int, n_lanes: int = 1,
                queue_depth_before: int = 0,
                stream_id: int = 0) -> None:
        from gpusim.trace.events import AtomicEvent
        self.atomic_events.append(AtomicEvent(
            cycle=cycle, sm_id=sm_id, warp_id=warp_id,
            kind=kind, op=op, space=space, line_addr=line_addr,
            latency=latency, n_lanes=n_lanes,
            queue_depth_before=queue_depth_before,
            stream_id=stream_id,
        ))

    def kernel_launch(self, *, stream_id: int, kernel_name: str,
                       grid: tuple, block: tuple,
                       launch_cycle: int, complete_cycle: int,
                       n_ctas: int) -> None:
        from gpusim.trace.events import KernelLaunch
        self.kernel_launch_events.append(KernelLaunch(
            stream_id=stream_id, kernel_name=kernel_name,
            grid=grid, block=block,
            launch_cycle=launch_cycle, complete_cycle=complete_cycle,
            n_ctas=n_ctas,
        ))
