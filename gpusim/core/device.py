from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from gpusim.config.schema import DeviceConfig


@dataclass
class DeviceRunResult:
    cycles: int
    outputs: dict[str, np.ndarray] = field(default_factory=dict)
    occupancy: dict[str, int] | None = None


class Device:
    def __init__(self, cfg: DeviceConfig, recorder=None):
        self.cfg = cfg
        self.n_sm = cfg.n_sm
        self.recorder = recorder

    def run(self, kernel, grid, block, params,
             regs_per_thread: int = 16, smem_per_cta: int = 0) -> DeviceRunResult:
        import numpy as np
        from gpusim.core.sm import SM
        from gpusim.core.exec import GlobalMemory, SharedMemory, ParamSpace
        from gpusim.core.cache.l2 import L2Cache
        from gpusim.core.hbm import HBM
        from gpusim.core.occupancy import compute_occupancy
        from gpusim.core.scheduler import make_cta_scheduler

        gmem = GlobalMemory()
        # SharedMemory pool sized for all CTAs across all SMs
        # (each CTA gets its own region keyed by cta_id, so capacity is just
        # max simultaneous CTAs * smem_per_cta)
        smem = SharedMemory(size_bytes=self.cfg.sm.smem_per_sm_bytes
                                          * max(self.n_sm, 1))
        p_dict: dict[str, int] = {}
        for name, val in params.items():
            if isinstance(val, np.ndarray):
                p_dict[name] = gmem.bind(name, val)
            else:
                p_dict[name] = int(val)
        paramspace = ParamSpace(p_dict)
        threads_per_cta = block[0] * block[1] * block[2]
        warps_per_cta = (threads_per_cta + 31) // 32
        occ = compute_occupancy(self.cfg.sm, threads_per_cta,
                                  regs_per_thread, smem_per_cta)

        hbm = HBM(self.cfg.hbm, recorder=self.recorder)
        l2 = L2Cache(self.cfg.cache, hbm, recorder=self.recorder)

        sms = []
        for i in range(self.n_sm):
            sm = SM(self.cfg.sm, sm_id=i, recorder=self.recorder, l2=l2, hbm=hbm)
            sm.initialize_for_run(kernel, gmem, smem, paramspace, grid, block, occ)
            sms.append(sm)

        cta_queue = []
        for cz in range(grid[2]):
            for cy in range(grid[1]):
                for cx in range(grid[0]):
                    cid = cx + cy * grid[0] + cz * grid[0] * grid[1]
                    cta_queue.append((cid, (cx, cy, cz)))
        scheduler = make_cta_scheduler(self.cfg.scheduler.cta_policy)
        cycle = 0
        cta_pointer = 0

        def _try_dispatch():
            nonlocal cta_pointer
            while cta_pointer < len(cta_queue):
                target_sm = scheduler.pick(sms, occ)
                if target_sm is None:
                    return
                cid, ctaid_xyz = cta_queue[cta_pointer]
                target_sm.activate_cta(
                    cid, ctaid_xyz, regs_per_thread, smem_per_cta,
                    threads_per_cta, warps_per_cta, cycle,
                )
                cta_pointer += 1
        _try_dispatch()

        while True:
            for sm in sms:
                sm.step_cycle(cycle)
            l2.tick(now=cycle)
            _try_dispatch()
            cycle += 1
            if (cta_pointer >= len(cta_queue)
                  and not any(sm.has_active_warps() for sm in sms)):
                break
            if cycle > 10_000_000:
                raise RuntimeError("simulation runaway > 1e7 cycles")

        outputs = {n: v for n, v in params.items() if isinstance(v, np.ndarray)}
        return DeviceRunResult(
            cycles=cycle, outputs=outputs,
            occupancy={"active_ctas": occ.active_ctas, "bottleneck": occ.bottleneck},
        )
