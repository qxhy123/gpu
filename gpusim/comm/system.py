"""Phase 10: Multi-GPU system orchestration."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class MultiGpuSystem:
    """N-GPU system. Phase 10."""
    gpus: list = field(default_factory=list)
    nvlink_fabric: object | None = None

    @classmethod
    def from_config(cls, cfg) -> "MultiGpuSystem":
        from gpusim.core.device import GPU
        n = getattr(cfg, "n_gpus", 1)
        gpus = [GPU(cfg, gpu_id=i) for i in range(n)]
        fabric = None
        if n > 1:
            try:
                from gpusim.comm.nvlink import NvlinkFabric
                fabric = NvlinkFabric.from_config(cfg, n_gpus=n)
            except ImportError:
                # NvlinkFabric created in T6 — graceful fallback for M1
                fabric = None
        return cls(gpus=gpus, nvlink_fabric=fabric)
