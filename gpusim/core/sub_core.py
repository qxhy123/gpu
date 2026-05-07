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
    recorder: object | None = None

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
        # bar.sync must drain the LSU (shared mem conflict cycles must complete)
        if instr.op == "bar.sync":
            if not self.fus.is_free(FUKind.LSU, now):
                return False, StallReason.STRUCTURAL
        # LSU outstanding queue
        if instr.op.startswith("ld.global.") or instr.op.startswith("st.global."):
            w.outstanding_loads = [c for c in w.outstanding_loads if c > now]
            if len(w.outstanding_loads) >= self.cfg.fu.lsu_outstanding:
                return False, StallReason.STRUCTURAL
        for r in _src_regs(instr):
            if w.scoreboard.has_pending(r, now):
                if w.scoreboard.origin_of(r) == "mem":
                    return False, StallReason.MEM_DEP
                return False, StallReason.SCOREBOARD
        kind = self.fus.classify(instr.op)
        if not self.fus.is_free(kind, now):
            return False, StallReason.STRUCTURAL
        return True, StallReason.ISSUED

    def step(self, now: int) -> list[StallReason]:
        self.scheduler.n = len(self.warps)
        states: list[StallReason] = [StallReason.IDLE] * len(self.warps)
        ready_flags: list[StallReason] = [StallReason.IDLE] * len(self.warps)
        for i, w in enumerate(self.warps):
            ok, why = self._is_ready(w, now)
            ready_flags[i] = why if not ok else StallReason.ISSUED
        chosen = self.scheduler.pick(now, candidates=lambda i: ready_flags[i] is StallReason.ISSUED)
        for i in range(len(self.warps)):
            states[i] = ready_flags[i]
        if self.recorder is not None:
            for i, w in enumerate(self.warps):
                self.recorder.warp_state(
                    cycle=now, warp_id=w.warp_id,
                    state=states[i].value,
                    pc=(w.stack.top().pc if w.stack and not w.stack.is_done() else -1),
                )
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
            if self.recorder is not None:
                self.recorder.instr_issue(
                    cycle=now, warp_id=w.warp_id, pc=instr.pc, op=instr.op,
                    src_loc=(instr.src_loc.file, instr.src_loc.line),
                    active_mask=w.fn_state.active_mask if w.fn_state else 0,
                )
            w.barrier_pc = w.stack.top().pc
            return
        if op == "bra":
            if self.recorder is not None:
                self.recorder.instr_issue(
                    cycle=now, warp_id=w.warp_id, pc=instr.pc, op=instr.op,
                    src_loc=(instr.src_loc.file, instr.src_loc.line),
                    active_mask=w.fn_state.active_mask if w.fn_state else 0,
                )
            target_pc = w.kernel.labels[instr.src[0]] if isinstance(instr.src[0], str) else 0
            if instr.pred is None:
                w.stack.update_top_pc(target_pc); w.stack.maybe_pop()
                return
            from gpusim.core.exec import _resolve_branch_mask
            taken_mask = _resolve_branch_mask(w.fn_state, instr)
            rpc = w.kernel.ipdom.get(w.stack.top().pc, target_pc)
            diverged = w.stack.diverge(taken_pc=target_pc, fallthrough_pc=w.stack.top().pc + 1,
                                       taken_mask=taken_mask, rpc=rpc)
            if diverged and self.recorder is not None:
                self.recorder.div_push(cycle=now, warp_id=w.warp_id,
                                       pc=instr.pc, rpc=rpc, taken_mask=taken_mask)
            if w.stack.maybe_pop() and self.recorder is not None:
                self.recorder.div_pop(cycle=now, warp_id=w.warp_id,
                                      pc=w.stack.top().pc if not w.stack.is_done() else -1)
            return
        # functional execution
        if self.recorder is not None:
            self.recorder.instr_issue(
                cycle=now, warp_id=w.warp_id, pc=instr.pc, op=instr.op,
                src_loc=(instr.src_loc.file, instr.src_loc.line),
                active_mask=w.fn_state.active_mask if w.fn_state else 0,
            )
        w.fn_state.active_mask = w.stack.top().active_mask
        w.fn_state.pc = w.stack.top().pc
        ex = w.executor if w.executor is not None else self.executor
        ex.execute(w.fn_state, instr)

        # determine issue occupancy / latency adjustments
        latency = self.fus.result_latency(op)
        if op.startswith(("ld.shared.", "st.shared.")):
            from gpusim.core.exec import shared_addresses_for_warp
            from gpusim.core.smem import bank_conflict_degree
            addrs = shared_addresses_for_warp(w.fn_state, instr)
            mask = w.fn_state.active_mask
            smem_conflict = bank_conflict_degree(
                addrs, active_mask=mask, banks=self.cfg.smem_banks)
            extra = smem_conflict - 1
            if extra > 0:
                kind = self.fus.classify(op)
                self.fus.reserve(kind, now, extra)
                latency += extra
            if self.recorder is not None:
                self.recorder.smem_access(
                    cycle=now, warp_id=w.warp_id,
                    conflict_degree=smem_conflict, addresses=addrs,
                )

        if op.startswith(("ld.global.", "st.global.")):
            from gpusim.core.exec import global_addresses_for_warp
            from gpusim.core.gmem import coalescing_info
            addrs = global_addresses_for_warp(w.fn_state, instr)
            info = coalescing_info(addrs, active_mask=w.fn_state.active_mask)
            w.last_gmem = info
            if self.recorder is not None:
                self.recorder.gmem_access(
                    cycle=now, warp_id=w.warp_id,
                    n_transactions=info.n_transactions, efficiency=info.efficiency,
                    addresses=addrs,
                )

        # operand collector bank conflict
        from gpusim.core.regfile import operand_extra_cycles
        srcs = _src_regs(instr)
        op_extra = operand_extra_cycles(srcs, banks=self.cfg.regfile.banks)
        if op_extra > 0:
            kind = self.fus.classify(op)
            self.fus.reserve(kind, now, op_extra)
            latency += op_extra
            w.last_operand_extra = op_extra

        # mark dst regs in scoreboard
        if latency > 0:
            origin = "mem" if op.startswith(("ld.global.", "ld.shared.", "ld.param.")) else "alu"
            for d in _dst_regs(instr):
                w.scoreboard.mark_write(d, now + latency, origin=origin)
        # register outstanding gmem load
        if op.startswith("ld.global."):
            w.outstanding_loads.append(now + latency)

        # advance PC
        w.stack.update_top_pc(w.stack.top().pc + 1)
        w.stack.maybe_pop()
