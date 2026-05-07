from __future__ import annotations
from dataclasses import dataclass
from gpusim.config.schema import SMConfig


@dataclass(frozen=True)
class OccupancyResult:
    active_ctas: int
    warps_per_cta: int
    max_by_warps: int
    max_by_regs: int
    max_by_smem: int
    bottleneck: str   # "warps" | "regs" | "smem" | "max_ctas_cap"


def compute_occupancy(cfg: SMConfig, threads_per_cta: int, regs_per_thread: int,
                      smem_per_cta: int) -> OccupancyResult:
    warps_per_cta = (threads_per_cta + 31) // 32
    by_warps = cfg.warps_per_sm // warps_per_cta if warps_per_cta else 0
    regs_per_cta = max(1, regs_per_thread * threads_per_cta)
    by_regs = cfg.regs_per_sm // regs_per_cta
    by_smem = cfg.smem_per_sm_bytes // max(1, smem_per_cta)
    raw = min(by_warps, by_regs, by_smem)
    active = min(raw, cfg.max_ctas_per_sm)
    if active == cfg.max_ctas_per_sm and raw >= cfg.max_ctas_per_sm:
        bn = "max_ctas_cap"
    else:
        if active == by_smem:
            bn = "smem"
        elif active == by_regs:
            bn = "regs"
        elif active == by_warps:
            bn = "warps"
        else:
            bn = "warps"
    return OccupancyResult(
        active_ctas=active, warps_per_cta=warps_per_cta,
        max_by_warps=by_warps, max_by_regs=by_regs, max_by_smem=by_smem,
        bottleneck=bn,
    )
