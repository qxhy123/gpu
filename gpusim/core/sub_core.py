from __future__ import annotations
from dataclasses import dataclass
from gpusim.config.schema import SMConfig
from gpusim.core.warp import Warp, StallReason
from gpusim.core.scheduler import LRRScheduler, GTOScheduler
from gpusim.core.functional_units import FUSet, FUKind
from gpusim.core.simt_stack import SIMTStack
from gpusim.core.exec import InstrExecutor
from gpusim.frontend.ir import Instr, Reg


def _make_scheduler(policy: str, n: int):
    if policy == "lrr": return LRRScheduler(n)
    if policy == "gto": return GTOScheduler(n)
    raise ValueError(f"unknown scheduler policy {policy!r}")


def _src_regs(instr: Instr) -> list[str]:
    out: list[str] = []
    for s in instr.src:
        if isinstance(s, Reg):
            out.append(s.name)
    if instr.pred is not None:
        out.append(instr.pred.reg)
    return out


def _dst_regs(instr: Instr) -> list[str]:
    return [d.name for d in instr.dst if isinstance(d, Reg)]


@dataclass
class SubCore:
    sub_core_id: int
    cfg: SMConfig
    executor: InstrExecutor
    warps: list[Warp]

    def __post_init__(self):
        self.fus = FUSet(self.cfg.fu)
        self.scheduler = _make_scheduler(self.cfg.scheduler.policy, len(self.warps))
        for w in self.warps:
            if w.stack is None:
                w.stack = SIMTStack(warp_size=32, entry_pc=0)

    def _is_ready(self, w: Warp, now: int) -> tuple[bool, StallReason]:
        if w.finished or w.stack is None or w.stack.is_done():
            return False, StallReason.IDLE
        if w.barrier_pc >= 0:
            return False, StallReason.BARRIER
        pc = w.stack.top().pc
        if pc >= len(w.kernel.instrs):
            w.finished = True
            return False, StallReason.IDLE
        instr = w.kernel.instrs[pc]
        for r in _src_regs(instr):
            if w.scoreboard.has_pending(r, now):
                return False, StallReason.SCOREBOARD
        kind = self.fus.classify(instr.op)
        if not self.fus.is_free(kind, now):
            return False, StallReason.STRUCTURAL
        return True, StallReason.ISSUED

    def step(self, now: int) -> list[StallReason]:
        states: list[StallReason] = [StallReason.IDLE] * len(self.warps)
        ready_flags: list[StallReason] = [StallReason.IDLE] * len(self.warps)
        for i, w in enumerate(self.warps):
            ok, why = self._is_ready(w, now)
            ready_flags[i] = why if not ok else StallReason.ISSUED
        chosen = self.scheduler.pick(now, candidates=lambda i: ready_flags[i] is StallReason.ISSUED)
        for i in range(len(self.warps)):
            states[i] = ready_flags[i]
        if chosen is None:
            return states
        w = self.warps[chosen]
        instr = w.kernel.instrs[w.stack.top().pc]
        kind = self.fus.classify(instr.op)
        occ = self.fus.issue_occupancy(instr.op)
        self.fus.reserve(kind, now, occ)
        self._issue(w, instr, now)
        states[chosen] = StallReason.ISSUED
        return states

    def _issue(self, w: Warp, instr: Instr, now: int) -> None:
        op = instr.op
        if op == "bar.sync":
            w.barrier_pc = w.stack.top().pc
            return
        if op == "bra":
            target_pc = w.kernel.labels[instr.src[0]] if isinstance(instr.src[0], str) else 0
            if instr.pred is None:
                w.stack.update_top_pc(target_pc); w.stack.maybe_pop()
                return
            from gpusim.core.exec import _resolve_branch_mask
            taken_mask = _resolve_branch_mask(w.fn_state, instr)
            rpc = w.kernel.ipdom.get(w.stack.top().pc, target_pc)
            w.stack.diverge(taken_pc=target_pc, fallthrough_pc=w.stack.top().pc + 1,
                            taken_mask=taken_mask, rpc=rpc)
            w.stack.maybe_pop()
            return
        w.fn_state.active_mask = w.stack.top().active_mask
        w.fn_state.pc = w.stack.top().pc
        self.executor.execute(w.fn_state, instr)
        latency = self.fus.result_latency(op)
        if latency > 0:
            for d in _dst_regs(instr):
                w.scoreboard.mark_write(d, now + latency)
        w.stack.update_top_pc(w.stack.top().pc + 1)
        w.stack.maybe_pop()
