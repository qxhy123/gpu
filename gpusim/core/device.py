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
        # T9 baseline: single-SM degenerate path. T10 will wire true multi-SM.
        from gpusim.core.sm import SM
        from gpusim.core.hbm import HBM
        from gpusim.core.cache.l2 import L2Cache
        hbm = HBM(self.cfg.hbm, recorder=self.recorder)
        l2 = L2Cache(self.cfg.cache, hbm, recorder=self.recorder)
        sm_cfg = self.cfg.sm
        sm_cfg._cache_for_run = self.cfg.cache
        sm_cfg._hbm_for_run = self.cfg.hbm
        sm = SM(sm_cfg, sm_id=0, recorder=self.recorder, l2=l2, hbm=hbm)
        res = sm.run(kernel=kernel, grid=grid, block=block, params=params,
                      regs_per_thread=regs_per_thread, smem_per_cta=smem_per_cta)
        return DeviceRunResult(
            cycles=res.cycles, outputs=res.outputs, occupancy=res.occupancy,
        )
