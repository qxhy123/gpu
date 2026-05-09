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
            sm.initialize_for_run(kernel, gmem, smem, paramspace, grid, block, occ,
                                    cluster_size=self.cfg.cluster_size)
            sms.append(sm)

        cta_queue = []
        for cz in range(grid[2]):
            for cy in range(grid[1]):
                for cx in range(grid[0]):
                    cid = cx + cy * grid[0] + cz * grid[0] * grid[1]
                    cta_queue.append((cid, (cx, cy, cz)))
        cluster_size = self.cfg.cluster_size
        grid_size = grid[0] * grid[1] * grid[2]
        if cluster_size > 1 and grid_size % cluster_size != 0:
            raise ValueError(
                f"cluster_size ({cluster_size}) must divide grid_size ({grid_size})"
            )

        scheduler = make_cta_scheduler(self.cfg.scheduler.cta_policy)
        cycle = 0
        cta_pointer = 0

        from gpusim.core.cluster import ClusterBarrierPool
        cluster_barriers: dict[int, ClusterBarrierPool] = {}
        for sm in sms:
            sm._device_cluster_barriers = cluster_barriers

        def _try_dispatch():
            nonlocal cta_pointer
            while cta_pointer < len(cta_queue):
                target_sms = scheduler.peek(sms, occ, k=cluster_size)
                if target_sms is None:
                    return
                scheduler.commit(k=cluster_size)
                cluster_id = cta_pointer // cluster_size
                if cluster_size > 1:
                    cluster_barriers[cluster_id] = ClusterBarrierPool(
                        expected=cluster_size,
                    )
                for i, sm in enumerate(target_sms):
                    cid, ctaid_xyz = cta_queue[cta_pointer + i]
                    sm.activate_cta(
                        cid, ctaid_xyz, regs_per_thread, smem_per_cta,
                        threads_per_cta, warps_per_cta, cycle,
                        cluster_id=cluster_id if cluster_size > 1 else -1,
                        cluster_rank=i if cluster_size > 1 else -1,
                    )
                    if self.recorder is not None:
                        self.recorder.cta_dispatch(
                            cycle=cycle, cta_id=cid, sm_id=sm.sm_id,
                            queue_position=cta_pointer + i,
                            active_warps_at_dispatch=sm.active_warp_count(),
                        )
                # Phase 5 cluster_dispatch event will be added in T19; skip if recorder lacks
                if cluster_size > 1 and self.recorder is not None:
                    if hasattr(self.recorder, "cluster_dispatch"):
                        self.recorder.cluster_dispatch(
                            cycle=cycle, cluster_id=cluster_id,
                            cluster_size=cluster_size,
                            sm_ids=tuple(sm.sm_id for sm in target_sms),
                            cta_ids=tuple(cta_queue[cta_pointer + i][0]
                                            for i in range(cluster_size)),
                            queue_position=cluster_id,
                        )
                cta_pointer += cluster_size
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
