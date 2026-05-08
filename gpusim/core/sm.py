from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from gpusim.config.schema import SMConfig
from gpusim.core.warp import Warp
from gpusim.core.simt_stack import SIMTStack
from gpusim.core.exec import (
    WarpFnState, GlobalMemory, SharedMemory, ParamSpace, InstrExecutor,
)
from gpusim.core.sub_core import SubCore
from gpusim.frontend.ir import Kernel


@dataclass
class SMRunResult:
    cycles: int
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    events: list[Any] = field(default_factory=list)
    occupancy: dict[str, int] | None = None


class SM:
    def __init__(self, cfg: SMConfig, recorder: object | None = None):
        self.cfg = cfg
        self.recorder = recorder

    def run(self, kernel, grid, block, params, regs_per_thread: int = 16,
            smem_per_cta: int = 0) -> SMRunResult:
        gmem = GlobalMemory()
        smem = SharedMemory(size_bytes=self.cfg.smem_per_sm_bytes)
        p_dict: dict[str, int] = {}
        for name, val in params.items():
            if isinstance(val, np.ndarray):
                p_dict[name] = gmem.bind(name, val)
            else:
                p_dict[name] = int(val)
        paramspace = ParamSpace(p_dict)

        threads_per_cta = block[0] * block[1] * block[2]
        warps_per_cta = (threads_per_cta + 31) // 32
        from gpusim.core.occupancy import compute_occupancy
        occ = compute_occupancy(self.cfg, threads_per_cta, regs_per_thread, smem_per_cta)

        cta_queue: list[tuple[int, tuple[int,int,int]]] = []
        for cz in range(grid[2]):
            for cy in range(grid[1]):
                for cx in range(grid[0]):
                    cid = cx + cy * grid[0] + cz * grid[0] * grid[1]
                    cta_queue.append((cid, (cx, cy, cz)))

        executor = InstrExecutor(kernel=kernel, gmem=gmem, smem=smem,
                                 params=paramspace, cta_id=0,
                                 ctaid=(0,0,0), nctaid=grid, ntid=block)

        from gpusim.core.cache.l1 import L1Cache
        from gpusim.core.cache.l2 import L2Cache
        from gpusim.core.hbm import HBM
        hbm = HBM(self.cfg.hbm)
        l2 = L2Cache(self.cfg.cache, hbm)
        l1 = L1Cache(self.cfg.cache, l2)

        sub_cores: list[SubCore] = [
            SubCore(i, self.cfg, executor, [], recorder=self.recorder, l1=l1)
            for i in range(self.cfg.sub_cores)
        ]

        active_warps: list[Warp] = []
        cycle = 0
        cta_pointer = 0

        def _activate_next_cta() -> bool:
            nonlocal cta_pointer
            if cta_pointer >= len(cta_queue): return False
            current_ctas = len({w.cta_id for w in active_warps})
            if current_ctas >= occ.active_ctas: return False
            cid, ctaid_xyz = cta_queue[cta_pointer]
            # Allocate at least smem_per_sm_bytes so shared-mem ops always have
            # a valid backing buffer, even when smem_per_cta is 0.
            alloc_bytes = smem_per_cta if smem_per_cta > 0 else self.cfg.smem_per_sm_bytes
            smem.allocate_cta(cid, alloc_bytes)
            cta_executor = InstrExecutor(kernel=kernel, gmem=gmem, smem=smem,
                                         params=paramspace, cta_id=cid,
                                         ctaid=ctaid_xyz, nctaid=grid, ntid=block)
            for wid_in_cta in range(warps_per_cta):
                fn = WarpFnState(warp_size=32, tids=tuple(range(wid_in_cta*32, wid_in_cta*32+32)))
                w = Warp(warp_id=cid * warps_per_cta + wid_in_cta, kernel=kernel,
                         fn_state=fn, stack=SIMTStack(warp_size=32, entry_pc=0),
                         cta_id=cid, executor=cta_executor)
                active_warps.append(w)
                sub_cores[w.warp_id % self.cfg.sub_cores].warps.append(w)
            if self.recorder is not None:
                self.recorder.cta_launch(cycle=cycle, cta_id=cid,
                                         warps=warps_per_cta,
                                         regs=regs_per_thread * threads_per_cta,
                                         smem_bytes=smem_per_cta)
            cta_pointer += 1
            return True

        while _activate_next_cta(): pass

        while True:
            for sc in sub_cores:
                sc.step(now=cycle)
            l1.install_completed_lines(now=cycle)

            by_cta: dict[int, list[Warp]] = {}
            for w in active_warps:
                by_cta.setdefault(w.cta_id, []).append(w)
            for cid, ws in by_cta.items():
                non_done = [w for w in ws if not w.finished]
                if non_done and all(w.barrier_pc >= 0 for w in non_done):
                    for w in non_done:
                        w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                        w.barrier_pc = -1

            retiring = []
            for cid, ws in by_cta.items():
                if all(w.finished or (w.stack and w.stack.is_done()) for w in ws):
                    retiring.append(cid)
            for cid in retiring:
                if self.recorder is not None:
                    self.recorder.cta_retire(cycle=cycle, cta_id=cid)
                smem.free_cta(cid)
                active_warps = [w for w in active_warps if w.cta_id != cid]
                for sc in sub_cores:
                    sc.warps = [w for w in sc.warps if w.cta_id != cid]
                _activate_next_cta()

            cycle += 1
            if cta_pointer >= len(cta_queue) and not active_warps:
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return SMRunResult(
            cycles=cycle, outputs=outputs, events=[],
            occupancy={"active_ctas": occ.active_ctas, "bottleneck": occ.bottleneck},
        )
