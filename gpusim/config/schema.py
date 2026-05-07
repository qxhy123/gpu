from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class SchedulerConfig:
    policy: str = "gto"


@dataclass
class RegFileConfig:
    banks: int = 4
    regs_per_subcore: int = 16384


@dataclass
class FUConfig:
    fp32_throughput: int = 1
    int32_throughput: int = 1
    lsu_throughput: int = 1
    bru_throughput: int = 1
    fp32_latency: int = 4
    int32_latency: int = 4
    fma_latency: int = 4
    bru_latency: int = 1
    smem_latency: int = 20
    gmem_latency: int = 400
    lsu_outstanding: int = 16


@dataclass
class SMConfig:
    sub_cores: int = 4
    warps_per_sm: int = 64
    threads_per_sm: int = 2048
    max_ctas_per_sm: int = 32
    regs_per_sm: int = 65536
    smem_per_sm_bytes: int = 48 * 1024
    smem_banks: int = 32
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    regfile: RegFileConfig = field(default_factory=RegFileConfig)
    fu: FUConfig = field(default_factory=FUConfig)
