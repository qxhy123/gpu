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
class CacheConfig:
    l1_size_bytes: int = 131072        # 128 KB
    l1_ways: int = 4
    l1_line_bytes: int = 128
    l1_hit_latency: int = 25
    l1_miss_check_latency: int = 5
    mshr_slots: int = 16
    l2_size_bytes: int = 4 * 1024 * 1024   # 4 MB
    l2_ways: int = 16
    l2_line_bytes: int = 128
    l2_hit_latency: int = 200
    l2_miss_install_latency: int = 10


@dataclass
class HBMConfig:
    channels: int = 8
    banks_per_channel: int = 16
    row_size_bytes: int = 4096
    row_hit_latency: int = 10
    row_miss_latency: int = 30


@dataclass
class TensorCoreConfig:
    tc_mma_latency: int = 8
    tc_mma_occupancy: int = 1
    tc_wgmma_latency: int = 32
    tc_wgmma_occupancy: int = 4
    wgmma_queue_capacity: int = 16


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
    cache: CacheConfig = field(default_factory=CacheConfig)        # NEW
    hbm: HBMConfig = field(default_factory=HBMConfig)              # NEW
    tensor_core: TensorCoreConfig = field(default_factory=TensorCoreConfig)
