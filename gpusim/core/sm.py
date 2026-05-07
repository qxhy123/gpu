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


class SM:
    def __init__(self, cfg: SMConfig):
        self.cfg = cfg

    def run(self, kernel: Kernel, grid: tuple[int,int,int], block: tuple[int,int,int],
            params: dict[str, np.ndarray | int]) -> SMRunResult:
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

        cycles_total = 0
        for cz in range(grid[2]):
          for cy in range(grid[1]):
            for cx in range(grid[0]):
                cta_id = cx + cy * grid[0] + cz * grid[0] * grid[1]
                smem.allocate_cta(cta_id, self.cfg.smem_per_sm_bytes)
                cycles_total += self._run_cta(kernel, gmem, smem, paramspace,
                                              cta_id, (cx,cy,cz),
                                              grid, block, warps_per_cta)
                smem.free_cta(cta_id)

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return SMRunResult(cycles=cycles_total, outputs=outputs)

    def _run_cta(self, kernel, gmem, smem, paramspace,
                 cta_id, ctaid, grid, block, warps_per_cta) -> int:
        executor = InstrExecutor(kernel=kernel, gmem=gmem, smem=smem,
                                 params=paramspace, cta_id=cta_id, ctaid=ctaid,
                                 nctaid=grid, ntid=block)
        all_warps: list[Warp] = []
        for wid in range(warps_per_cta):
            tids = tuple(range(wid*32, wid*32+32))
            fn = WarpFnState(warp_size=32, tids=tids)
            all_warps.append(Warp(warp_id=wid, kernel=kernel, fn_state=fn,
                                  stack=SIMTStack(warp_size=32, entry_pc=0),
                                  cta_id=cta_id))
        groups: list[list[Warp]] = [[] for _ in range(self.cfg.sub_cores)]
        for w in all_warps:
            groups[w.warp_id % self.cfg.sub_cores].append(w)
        sub_cores = [SubCore(i, self.cfg, executor, groups[i])
                     for i in range(self.cfg.sub_cores)]

        cycle = 0
        while True:
            for sc in sub_cores:
                sc.step(now=cycle)
            non_done = [w for w in all_warps if not w.finished]
            if non_done and all(w.barrier_pc >= 0 for w in non_done):
                for w in non_done:
                    w.stack.update_top_pc(w.barrier_pc + 1); w.stack.maybe_pop()
                    w.barrier_pc = -1
            cycle += 1
            if all(w.finished or (w.stack and w.stack.is_done()) for w in all_warps):
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")
        return cycle
