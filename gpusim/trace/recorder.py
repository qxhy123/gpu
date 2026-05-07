from __future__ import annotations
from collections import defaultdict
from typing import Iterator
from .events import (
    WarpStateSegment, InstrIssueEvent, SmemEvent, GmemEvent,
    DivEvent, CtaEvent, BarEvent,
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

    def instr_issue(self, *, cycle, warp_id, pc, op, src_loc, active_mask) -> None:
        self._instr_issues.append(InstrIssueEvent(
            cycle=cycle, warp_id=warp_id, pc=pc, op=op,
            src_loc=tuple(src_loc), active_mask=int(active_mask)))

    def instr_issues(self) -> list[InstrIssueEvent]:
        return list(self._instr_issues)

    def smem_access(self, *, cycle, warp_id, conflict_degree, addresses) -> None:
        self._smem.append(SmemEvent(cycle, warp_id, conflict_degree, tuple(addresses)))

    def smem_accesses(self) -> list[SmemEvent]:
        return list(self._smem)

    def gmem_access(self, *, cycle, warp_id, n_transactions, efficiency, addresses) -> None:
        self._gmem.append(GmemEvent(cycle, warp_id, n_transactions, float(efficiency), tuple(addresses)))

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

    def bar_reach(self, *, cycle, cta_id, barrier_id=0) -> None:
        self._bar.append(BarEvent("REACH", cycle, cta_id, barrier_id))

    def bar_release(self, *, cycle, cta_id, barrier_id=0) -> None:
        self._bar.append(BarEvent("RELEASE", cycle, cta_id, barrier_id))

    def bar_events(self) -> list[BarEvent]:
        return list(self._bar)
