# gpusim

Teaching-oriented NVIDIA GPU microarchitecture simulator.

## Quick start
```bash
pip install -e ".[dev]"   # 自动安装 ml_dtypes>=0.4（Phase 3 依赖）
gpusim doctor
python examples/vector_add/run.py
```

## What you can learn
- SIMT execution + branch divergence (`examples/divergence_demo/`)
- Global memory coalescing (`examples/coalescing_demo/`)
- Shared memory bank conflicts (`examples/bank_conflict_demo/`)
- Reduction with shared memory + bar.sync (`examples/reduction_smem/`)
- Tiled matmul (`examples/tiled_matmul/`)
- Occupancy bottlenecks (`docs/tutorial/06-occupancy.md`)
- Cache hierarchy (L1, L2, HBM) — `examples/l1_thrash_demo/` + Chapter 08
- Shared memory vs L1 cache — `examples/smem_vs_l1_demo/` + Chapter 09
- HBM bandwidth saturation — `examples/bw_saturation_demo/` + Chapter 10
- DRAM row buffer locality — `examples/row_buffer_demo/` + Chapter 11
- **Tensor Core (sync mma)** — 3 shapes × 6 precisions (FP16/BF16/FP8-E4M3/FP8-E5M2/TF32/INT8) — `examples/tc_matmul_precisions/` + Chapter 12
- **混合精度累加** — FP8 输入 × FP32 累加 — `examples/mixed_accum/` + Chapter 13
- **Hopper wgmma (异步 warp-group MMA)** — m64n128k16 形状 — `examples/wgmma_basic/` + Chapter 14
- **TMA-lite + wgmma 异步流水线** — cp.async.bulk + mbarrier — `examples/wgmma_async_pipeline/` + Chapter 15
- **Multi-SM topology** — 8 SMs (default, configurable via `DeviceConfig.n_sm`), CTA→SM scheduler (RR / greedy) — `examples/multi_sm_scheduler/` + Chapter 16
- **Cross-SM L2 sharing** — shared L2 with cross-SM MSHR coalescing — `examples/l2_sharing_demo/` + Chapter 17
- **TMA store pipeline** — `cp.async.bulk.tensor.2d.global.shared::cta` + commit_group / wait_group — `examples/tma_store_matmul/` + Chapter 18
- **Hopper Cluster (CGA) + distributed shared memory (dsmem)** — `mapa.shared::cluster` + `ld/st.shared::cluster` for cross-CTA smem access — `examples/cluster_basic/` + Chapter 19
- **Cluster barrier sync** — `barrier.cluster.{arrive,wait}` two-phase async cluster sync + `mbarrier.{init,arrive,try_wait}.shared::cluster` — `examples/cluster_matmul_dsmem/` + Chapter 20
- **Cluster TMA load** — `cp.async.bulk.tensor.shared::cluster` writes to remote CTA's smem — `examples/cluster_tma_pipeline/` + Chapter 21
- **Global memory atomics (gmem atomic)** — 5 ops (add/min/max/exch/cas) × 3 dtypes (u32/s32/f32), L2AtomicQueue per-line FIFO for cross-SM serialization — `examples/atom_histogram/` + Chapter 22
- **Shared memory atomics (smem atomic)** — same 5 ops × 3 dtypes, bank-conflict-aware serialization — `examples/atom_reduction_smem/` + Chapter 23
- **Cluster TMA store + cooperative epilogue** — `cp.async.bulk.tensor.2d.global.shared::cluster` cluster-scope async epilogue — `examples/cluster_cooperative_epilogue/` + Chapter 24
- **CAS-based spinlock** — `atom.cas` for lock-free synchronization patterns — `examples/atom_cas_spinlock/` + Chapter 25
- **Reduction min/max** — `atom.red.min/max` across SMs, 4 new metrics (atomic_throughput_per_line, serialization_overhead, atom_red_ratio, cooperative_overlap) — `examples/red_min_max/` + Chapter 26
- **Multi-stream concurrency** — `gpusim.Stream` + `Stream.launch` + `gpusim.synchronize()` multi-stream API, round-robin CTA scheduler across streams, 4 stream metrics — `examples/concurrent_vector_add_2stream/` + Chapter 27
- **Compute vs memory overlap** — interleave compute-heavy and memory-heavy kernels across streams — `examples/compute_vs_memory_overlap/` + Chapter 28
- **L2/HBM contention across streams** — model cross-stream L2 bandwidth pressure — `examples/l2_contention_2stream/` + Chapter 29
- **Stream scheduler fairness** — serial vs concurrent scheduling, Jain fairness index — `examples/stream_priority_serial_vs_concurrent/` + Chapter 30
- **True concurrent scheduling** — `ConcurrentStreamScheduler` per-cycle weighted round-robin across streams — `examples/true_concurrent_overlap/` + Chapter 31
- **Stream priority** — high/normal/low priority levels with 4:2:1 dispatch weights — `examples/priority_demo/` + Chapter 32
- **CUDA Events** — `Event` class + `Stream.record`/`wait` for cross-stream synchronization + `StreamEvent` Perfetto annotations — `examples/event_producer_consumer/` + `examples/event_fanout/` + Chapters 33–34
- **L2 set-window partitioning** — `Stream.set_l2_window` / `cudaStreamAttributeAccessPolicyWindow` equivalent; API + L2 registration + eviction enforcement (Phase 9); `l2_window_hit_rate` + `l2_eviction_protected_count` metrics — `examples/l2_window_demo/` + Chapter 35
- **Multi-stream full pipeline** — load → compute → store across 3 streams with event ordering — `examples/multi_stream_pipeline_full/` + Chapter 36
- **Per-cycle scheduler + real overlap** — `Device.run_streams` per-cycle main loop with actual cross-grid CTA interleave; `actual_cross_grid_overlap_cycles` metric — `examples/phase8_overlap_real/` + Chapter 37
- **Multi-event fan-in** — `Stream.wait_all([events])` AND-semantics barrier for multiple upstream dependencies — `examples/multi_event_fan_in/` + Chapter 38
- **Event timing benchmark** — `Event.elapsed_time(start, end)` cycle-delta profiling utility; `l2_eviction_protected_count` metric — `examples/event_timing_benchmark/` + Chapter 39
- **Multi-GPU system setup** — `cfg.n_gpus` configurable, `MultiGpuSystem` wrapping N GPUs sharing `NvlinkFabric`, point-to-point NVLink links — `examples/multi_gpu_setup/` + Chapter 40
- **Ring allreduce** — bandwidth-optimal ring algorithm for large messages, `nvlink_bandwidth_utilization` + `per_rank_communication_volume` metrics — `examples/ring_allreduce/` + Chapter 41
- **Tree allreduce** — latency-optimal tree algorithm for small messages, `algo_efficiency_ring_vs_tree` metric, auto-pick at 4096-byte threshold — `examples/tree_allreduce/` + Chapter 42
- **DDP training step** — end-to-end allreduce + broadcast pipeline mimicking DistributedDataParallel gradient sync, `collective_op_breakdown` metric — `examples/ddp_training_step/` + Chapter 43

## Run a kernel and inspect the report
```bash
# Option 1: Python API — Phase 8 multi-stream example
python -c "
import numpy as np, gpusim, pathlib
from gpusim.config.loader import load_default

cfg = load_default()
cfg.n_sm = 8
n = 1024
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c0 = np.zeros(n, dtype=np.float32)
c1 = np.zeros(n, dtype=np.float32)
ptx = pathlib.Path('examples/vector_add/kernel.ptx').read_text()

# Phase 8: multi-stream API with priority + events
from gpusim import Event
s0 = gpusim.Stream(priority='high')
s1 = gpusim.Stream(priority='normal')
s0.launch(ptx, grid=(4,1,1), block=(128,1,1),
          params={'A':a,'B':b,'C':c0,'N':n}, kernel_name='vadd_s0', config=cfg)
ev = Event()
s0.record(ev)        # record event after s0 kernel
s1.wait(ev)          # s1 waits for s0 to finish before launching
s1.launch(ptx, grid=(4,1,1), block=(128,1,1),
          params={'A':b,'B':a,'C':c1,'N':n}, kernel_name='vadd_s1', config=cfg)

multi_res = gpusim.synchronize([s0, s1], config=cfg)
print(multi_res.stream_summary())
print(multi_res.cross_stream_concurrency_gain())
print(multi_res.priority_dispatch_share())
print(multi_res.event_wait_cycles())
print(multi_res.event_chain_critical_path())
print(multi_res.l2_window_hit_rate())
# Phase 8: 6 stream metrics (4 from Phase 7 + 2 new)
from gpusim.analysis.metrics import (
    stream_concurrency_factor, compute_memory_overlap,
    l2_bandwidth_per_stream, stream_fairness_jain,
)

# Phase 6: single-stream API still fully supported (100% backward compatible)
res = gpusim.run(ptx_src=ptx, grid=(8,1,1), block=(128,1,1),
                 params={'A':a,'B':b,'C':c0,'N':n}, mode='timing', config=cfg)
res.html_report('report.html')
res.perfetto('trace.json')
print(res.summary())
print(res.atomic_summary())
print(res.atomic_metrics)         # dict: atomic_throughput_per_line, serialization_overhead, atom_red_ratio, cooperative_overlap
print(res.cluster_summary())
print(res.device_summary())
print(res.cta_dispatch_events_df)
# Phase 3: Tensor Core metrics (available when kernel uses mma/wgmma)
# print(res.tc_summary())
# print(res.tc_metrics)           # dict with per-precision MMA counts
# res.mma_events_df()             # DataFrame of sync-mma events
# res.wgmma_events_df()           # DataFrame of async wgmma events
# res.tma_events_df()             # DataFrame of TMA bulk-copy events
# res.mbarrier_events_df()        # DataFrame of mbarrier phase transitions
# res.atomic_events_df()          # DataFrame of AtomicEvent trace events
"

# Option 2: CLI (after staging .npy inputs)
python -c "import numpy as np; np.save('a.npy', np.random.randn(1024).astype(np.float32)); np.save('b.npy', np.random.randn(1024).astype(np.float32)); np.save('c.npy', np.zeros(1024, dtype=np.float32))"
gpusim run examples/vector_add/kernel.ptx \
    --grid 8 --block 128 \
    --inputs A:a.npy,B:b.npy,C:c.npy,N:1024 \
    --mode timing \
    --output report.html --perfetto trace.json
```
- `report.html` — open in any browser; Phase 3 adds §11 Tensor Core、§12 wgmma、§13 TMA、§14 Barrier sections; Phase 4 adds §15–§18 (CTA dispatch、L2 MSHR、bulk-store overlap、per-SM utilization); Phase 5 adds §19–§20 (cluster dispatch、cluster barrier stats); Phase 6 adds §21 Atomic Operations、§22 Cooperative Epilogue; Phase 7 adds §27 Stream Concurrency、§28 Per-Stream Breakdown; Phase 8 adds §29 Cross-Stream Concurrency Gain、§30 Priority Dispatch、§31 Event/L2-Window Stats; Phase 9 adds §32 Actual Overlap Cycles + L2 Eviction Protection; Phase 10 adds §33 NVLink Bandwidth + §34 Collective Operations
- `trace.json` — drag into https://ui.perfetto.dev; Phase 3 adds TC / TMA / Barrier tracks; Phase 4 adds per-SM swimlane; Phase 5 adds cluster swimlane; Phase 6 adds Atomic track (AtomicEvent per-line FIFO serialization); Phase 7 adds Stream-N swimlanes (one per stream_id); Phase 8 adds StreamEvent annotations (record/wait markers per stream with cycle timestamps); Phase 9 adds Perfetto async arrows for record→wait event flow; Phase 10 adds NVLink swimlane (per GPU→GPU link) + Collective swimlane (per rank)

## What's modeled

### Phase status

| Phase | Status | Highlights |
|-------|--------|-----------|
| 1 | ✅ done | SIMT core, bank conflicts, coalescing, multi-CTA occupancy |
| 2 | ✅ done | Tag-precise L1/L2/HBM cache hierarchy |
| 3 | ✅ done | Tensor Core (sync mma + wgmma) + TMA-lite |
| 4 | ✅ done | Multi-SM topology, CTA scheduler, shared L2, TMA store |
| 5 | ✅ done | Hopper Cluster CGA, dsmem, cluster barrier, cluster TMA |
| 6 | ✅ done | gmem/smem atomics (5 ops × 3 dtypes), cluster TMA store, cooperative epilogue |
| 7 | ✅ done | Multi-stream API (Stream/launch/synchronize), 4 stream metrics, KernelLaunch trace, §27/§28 HTML, Stream-N Perfetto swimlanes |
| 8 | ✅ done | True concurrent scheduler (ConcurrentStreamScheduler), stream priority (4:2:1 weights), CUDA Events, L2 set-window API, 6 metrics, §29/§30/§31 HTML, StreamEvent Perfetto |
| 9 | ✅ done | Per-cycle Device.run_streams, L2 eviction window protection, Stream.wait_all, Event.elapsed_time, 2 metrics, §32 HTML, Perfetto async arrows, 3 examples (36–38), tutorials 37–39 |
| 10 | ✅ done | Multi-GPU (cfg.n_gpus), MultiGpuSystem, NVLink fabric, Comm (NCCL-equivalent): ring + tree allreduce, broadcast, allgather; 4 metrics, 2 trace events, §33/§34 HTML, Perfetto NVLink + Collective swimlanes, 4 examples (39–42), tutorials 40–43 |
| 11+ | future | Warp shuffle, ITS, full per-cycle CTA slicing (grid-level interleave) |

### Phase 1 ✅ — SIMT 基础
Single SM, cycle-approximate, Hopper-shaped. PTX subset (~30 ops). Shared memory bank conflicts, global memory coalescing, regfile bank conflicts, multi-CTA occupancy.

### Phase 2 ✅ — 存储层次
**Cache hierarchy: tag-precise L1 (128 KB / 4-way / 16 MSHR), L2 (4 MB / 16-way / write-back), HBM (8 channels × 16 banks + row buffer + channel queue).** Cache hit rates, HBM bandwidth utilization, row buffer locality, write-back traffic all surfaced in the HTML report and Python API.

### Phase 3 ✅ — Tensor Core + wgmma + TMA-lite
**Sync mma:** 3 matrix shapes (m16n8k8 / m16n8k16 / m16n8k32) × 6 数据精度 (FP16, BF16, FP8-E4M3, FP8-E5M2, TF32, INT8)，周期精确吞吐建模，每次 mma 写入 `tc_metrics`。**Hopper wgmma:** 异步 warp-group MMA，形状 m64n128k16，跨 warp 累加器寄存器。**TMA-lite:** `cp.async.bulk.tensor.2d` + mbarrier arrive/wait，全流水线异步内存搬运。HTML 报告新增 §11–§14；Perfetto 新增 TC / TMA / Barrier 专用 track；新增依赖 `ml_dtypes>=0.4`。

### Phase 4 ✅ — Multi-SM + CTA Scheduler + L2 Sharing + TMA Store
**Multi-SM topology:** default 8 SMs, configurable via `DeviceConfig.n_sm`. **CTA→SM scheduler:** round-robin (`rr`) and greedy policies, selectable via `cfg.scheduler.cta_policy`. **Shared L2 with cross-SM MSHR coalescing:** requests from multiple SMs merge in the shared L2 MSHR, reducing HBM traffic when SMs access overlapping cache lines. **TMA store:** `cp.async.bulk.tensor.2d.global.shared::cta` + `commit_group` / `wait_group` for async smem→gmem transfers. HTML report adds §15–§18; Perfetto adds per-SM swimlane.

### Phase 5 ✅ — Hopper Cluster (CGA) + Distributed Shared Memory + Cluster TMA
**Hopper Thread-Block Cluster (CGA):** groups of 2/4/8 CTAs share a cluster-scoped address space. **Distributed shared memory (dsmem):** `mapa.shared::cluster` maps a remote CTA's smem address; `ld.shared::cluster` / `st.shared::cluster` perform cross-CTA smem reads/writes. **Two-phase cluster barrier:** `barrier.cluster.arrive` + `barrier.cluster.wait` provide hardware-accelerated cross-CTA synchronization. **Cluster mbarrier:** `mbarrier.init.shared::cluster`, `mbarrier.arrive.shared::cluster`, `mbarrier.try_wait.shared::cluster` extend async-pipeline barriers to cluster scope. **Cluster TMA load:** `cp.async.bulk.tensor.shared::cluster` issues a TMA load that writes directly into a remote CTA's smem, enabling zero-copy distribution of tiles across the cluster. HTML report adds §19–§20 (cluster dispatch、cluster barrier stats); Perfetto adds cluster swimlane.

### Phase 6 ✅ — Atomics + Cluster TMA Store + Cooperative Epilogue
**Global memory atomics (gmem atomic):** `atom.global` and `red.global` supporting 5 operations (add / min / max / exch / cas) × 3 data types (u32 / s32 / f32). **L2AtomicQueue:** per-cache-line FIFO queue that serializes concurrent atomic requests from multiple SMs, modeling real hardware cross-SM contention. **Shared memory atomics (smem atomic):** `atom.shared` and `red.shared` with bank-conflict-aware serialization. **Cluster TMA store:** `cp.async.bulk.tensor.2d.global.shared::cluster` cluster-scoped async smem→gmem epilogue. **Cooperative epilogue:** cluster-coordinated writeback pattern. **4 new metrics:** `atomic_throughput_per_line`, `serialization_overhead`, `atom_red_ratio`, `cooperative_overlap`. **1 new trace event:** `AtomicEvent` (op, dtype, address, SM, cycle) recorded to Parquet + surfaced in Perfetto Atomic track. **HTML report adds §21** (Atomic Operations — throughput, contention, per-op breakdown) **and §22** (Cooperative Epilogue — cluster store overlap).

### Phase 7 ✅ — Multi-Stream / Multi-Kernel Concurrency
**Multi-stream API:** `gpusim.Stream` dataclass + `Stream.launch(ptx, grid, block, params, kernel_name, config)` enqueues kernels; `gpusim.synchronize(streams, config)` drains all streams and returns a `MultiStreamResult`. **100% backward compatible:** `gpusim.run()` is unchanged. **Round-robin CTA scheduler across streams:** `MultiStreamScheduler` dispatches CTAs from multiple streams in round-robin order, propagating `stream_id` through all dispatch events. **1 new trace event:** `KernelLaunch` (stream_id, kernel_name, grid, block, launch_cycle, complete_cycle, n_ctas) recorded to Parquet. **stream_id propagated to 11 existing events:** CTA dispatch, warp events, SubCore events and more now carry `stream_id` for per-stream attribution. **4 new analysis metrics:** `stream_concurrency_factor`, `compute_memory_overlap`, `l2_bandwidth_per_stream`, `stream_fairness_jain` (Jain's fairness index). **`MultiStreamResult` API:** `stream_summary()`, `stream_metrics`, `kernel_launch_events_df`, `per_stream_events_df`, `fairness()`, `overlap_ratio()`. **HTML report adds §27** (Stream Concurrency — concurrency factor, timeline) **and §28** (Per-Stream Breakdown — cycles, CTAs, bandwidth per stream). **Perfetto adds Stream-N swimlanes:** one process track per unique `stream_id` shows kernel durations across streams.

### Phase 8 ✅ — True Concurrent Scheduler + Priority + Events + L2 Window
**True concurrent scheduler:** `ConcurrentStreamScheduler` replaces `MultiStreamScheduler`; per-cycle weighted round-robin dispatches CTAs across streams each tick, maximizing CTA-level interleave within `Device.run`'s per-grid execution loop. (`MultiStreamScheduler` kept as a backward-compatible alias.) **Note (resolved in Phase 9):** full per-cycle grid-level interleave (slicing `Device.run` by cycle) ships in Phase 9 M1. **Stream priority:** `Stream(priority='high'|'normal'|'low')` with configurable dispatch weights (default 4:2:1 high/normal/low). **1 new metric:** `priority_dispatch_share` (dict of priority → fraction of CTAs dispatched). **CUDA Events:** `gpusim.Event` class + `Stream.record(event)` + `Stream.wait(event)` for cross-stream ordering. **1 new trace event:** `StreamEvent` (event_id, stream_id, kind='record'|'wait', cycle) recorded to Parquet + emitted as Perfetto instant events on per-stream tracks. **2 new metrics:** `event_wait_cycles` (dict of event_id → cycles a stream stalled waiting), `event_chain_critical_path` (longest event-dependency chain in cycles). **L2 set-window partitioning:** `Stream.set_l2_window(hit_ratio, num_bytes, action)` — `cudaStreamAttributeAccessPolicyWindow` equivalent; registers a priority window on the L2 cache (API + L2 registration + window-aware eviction logic in `L2Cache.register_stream_window`). **2 new metrics:** `l2_window_hit_rate` (fraction of accesses hitting in-window lines), `l2_window_protection_efficiency` (fraction of window lines not evicted). **Note (resolved in Phase 9):** per-access hit/in_window data plumbing (GmemEvent fields + CacheSet.install enforcement) completes in Phase 9 M2, unblocking the `l2_window_hit_rate` metric denominator. **HTML report adds §29** (Cross-Stream Concurrency Gain), **§30** (Priority Dispatch Share), **§31** (Event Chain + L2 Window Stats). **Perfetto adds StreamEvent annotations** per stream. **6 new examples** (examples 30–35) + **6 tutorial chapters 31–36**. **100% backward compatible:** Phase 1–7 APIs unchanged.

### Phase 9 ✅ — Per-Cycle Scheduler + L2 Eviction + Multi-Event Wait + Event Timing
**Per-cycle Device.run_streams main loop (M1):** `Device.run_streams` rewritten as a per-cycle tick loop — each cycle it checks all streams for ready CTAs and dispatches them in weighted round-robin order, enabling true cross-grid CTA interleave. `actual_cross_grid_overlap_cycles` metric counts cycles where CTAs from ≥ 2 grids are simultaneously in-flight. (Full per-cycle CTA slicing of individual `Device.run` calls is Phase 10+.) **L2 eviction integration (M2):** `CacheSet.install` now enforces window-protection: lines belonging to a stream's L2 window are shielded from eviction by other streams' accesses. `GmemEvent` gains `hit` and `in_window` boolean fields, wiring the per-access data needed to compute `l2_window_hit_rate` correctly. **Stream.wait_all (M3):** `Stream.wait_all(events: list[Event])` provides AND-semantics fan-in — the stream stalls until every event in the list has been recorded, enabling multi-producer → single-consumer patterns. **Event.elapsed_time (M3):** `Event.elapsed_time(start_event, end_event)` static utility returns the cycle delta between two recorded events, mirroring `cudaEventElapsedTime`. **2 new metrics:** `actual_cross_grid_overlap_cycles` (int, cycles of real cross-grid concurrency), `l2_eviction_protected_count` (int, cache-line installs protected by window). **HTML §32** — Combined Overlap section: actual overlap cycles vs. estimated, L2 eviction protection count. **Perfetto async arrows** for record→wait flow: each `Stream.record` / `Stream.wait` pair generates a Perfetto async slice so the event dependency is visible as an arrow in the trace viewer. **3 new examples** (36–38): `phase8_overlap_real`, `multi_event_fan_in`, `event_timing_benchmark`. **3 new tutorial chapters** (37–39). **100% backward compatible:** Phase 1–8 APIs unchanged.

### Phase 10 ✅ — Multi-GPU + NVLink + NCCL-equivalent Collectives
**cfg.n_gpus configurable (default 1 = backward compatible):** `DeviceConfig.n_gpus` sets the number of simulated GPUs; existing single-GPU code requires no changes. **MultiGpuSystem:** wraps N `GPU` instances sharing a single `NvlinkFabric`, providing `.run_all()` for multi-GPU kernel dispatch. **NVLink fabric:** point-to-point links modeled with per-link bandwidth + latency; `NvlinkFabric.build_all_to_all(n_gpus)` constructs the default fully-connected topology. **Comm class (NCCL-equivalent):** `Comm(gpus, fabric)` carries `rank` + `world_size`; exposes `allreduce(buf, op)`, `broadcast(buf, root)`, `allgather(sendbuf) → recvbuf`. **Allreduce algorithms:** ring (bandwidth-optimal, large messages) and tree (latency-optimal, small messages) with auto-pick at the 4096-byte threshold; selectable via `algo='ring'|'tree'|'auto'`. **4 new metrics:** `nvlink_bandwidth_utilization`, `collective_op_breakdown`, `algo_efficiency_ring_vs_tree`, `per_rank_communication_volume`. **2 new trace events:** `NvlinkTransfer` (src_gpu, dst_gpu, nbytes, start_cycle, end_cycle) + `CollectiveOp` (op, algo, world_size, nbytes, start_cycle, end_cycle). **HTML report adds §33** (NVLink Bandwidth — per-link utilization + transfer timeline) **and §34** (Collective Operations — op breakdown, algo efficiency, per-rank volume). **Perfetto adds NVLink swimlane** (one track per GPU→GPU link showing NvlinkTransfer slices) **and Collective swimlane** (one track per rank showing CollectiveOp slices). **4 new examples (39–42):** `multi_gpu_setup`, `ring_allreduce`, `tree_allreduce`, `ddp_training_step`. **4 new tutorial chapters (40–43). 100% backward compatible:** Phase 1–9 APIs unchanged.

## What's NOT modeled
Warp shuffle, ITS, full per-cycle CTA slicing of individual grid execution (Phase 9 M1 adds cross-grid interleave; intra-grid slicing is Phase 11+). See `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` section 11.

## Tutorial
Read `docs/tutorial/00-intro.md` first.

| Chapter | 主题 |
|---------|------|
| 00 | Introduction |
| 01 | SIMT execution |
| 02 | Warp scheduler |
| 03 | Memory coalescing |
| 04 | Bank conflicts |
| 05 | Branch divergence |
| 06 | Occupancy |
| 07 | Tiled matmul |
| 08 | Cache hierarchy |
| 09 | Shared memory vs L1 |
| 10 | HBM bandwidth |
| 11 | DRAM row buffer |
| **12** | **Tensor Core 入门 — sync mma 形状与精度** |
| **13** | **精度权衡 — FP8/BF16/TF32/INT8** |
| **14** | **混合精度累加器** |
| **15** | **wgmma + TMA 异步流水线** |
| **16** | **Multi-SM 调度 — CTA→SM 分配策略 (RR / greedy)** |
| **17** | **L2 sharing — 跨 SM MSHR 合并与命中率** |
| **18** | **TMA store 流水线 — commit_group / wait_group** |
| **19** | **Hopper Cluster CGA 入门 — mapa / ld.shared::cluster / cluster barrier** |
| **20** | **Cluster + wgmma + dsmem — 跨 CTA smem 分块矩阵乘** |
| **21** | **Cluster TMA 流水线 — cp.async.bulk.tensor.shared::cluster** |
| **22** | **全局内存原子操作 — L2AtomicQueue 跨 SM 序列化** |
| **23** | **共享内存原子操作 — bank conflict 感知序列化** |
| **24** | **Cluster TMA store + 协作尾写 (cooperative epilogue)** |
| **25** | **CAS 自旋锁 — lock-free 同步模式** |
| **26** | **原子 reduction vs 共享内存 reduction — 吞吐比较** |
| **27** | **Multi-stream 并发基础 — Stream / launch / synchronize API** |
| **28** | **计算与内存重叠 — compute_memory_overlap 跨流流水线** |
| **29** | **L2/HBM 跨流竞争 — l2_bandwidth_per_stream 带宽压力** |
| **30** | **调度器公平性 — stream_fairness_jain Jain 公平指数** |
| **31** | **真并发调度 — ConcurrentStreamScheduler 加权轮询与 cross_stream_concurrency_gain** |
| **32** | **流优先级 — high/normal/low 权重 4:2:1 + priority_dispatch_share** |
| **33** | **CUDA Events 入门 — Event / record / wait + event_wait_cycles** |
| **34** | **Event 链路径 — event_chain_critical_path + 生产者消费者模式** |
| **35** | **L2 set-window 分区 — cudaStreamAttributeAccessPolicyWindow API + 命中率指标** |
| **36** | **多流全流水线 — load → compute → store 跨流 event 排序** |
| **37** | **Per-cycle 调度器与真实重叠 — Device.run_streams 逐周期主循环 + actual_cross_grid_overlap_cycles** |
| **38** | **多事件 fan-in — Stream.wait_all AND 语义屏障** |
| **39** | **Event 计时与性能分析 — Event.elapsed_time 周期差 + l2_eviction_protected_count** |
| **40** | **Multi-GPU 系统搭建 — cfg.n_gpus + MultiGpuSystem + NvlinkFabric 点对点拓扑** |
| **41** | **Ring Allreduce — 带宽最优环形算法 + nvlink_bandwidth_utilization + per_rank_communication_volume** |
| **42** | **Tree Allreduce — 延迟最优树形算法 + algo_efficiency_ring_vs_tree + 4096 字节自动选择阈值** |
| **43** | **DDP 训练步骤 — allreduce + broadcast 梯度同步流水线 + collective_op_breakdown** |
