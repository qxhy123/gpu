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

## Run a kernel and inspect the report
```bash
# Option 1: Python API — Phase 5 cluster example
python -c "
import numpy as np, gpusim, pathlib
from gpusim.config.loader import load_default

cfg = load_default()
cfg.n_sm = 8                    # Phase 4 default
cfg.scheduler.cta_policy = 'greedy'
cfg.cluster_size = 4              # Phase 5: 4-CTA clusters
n = 1024
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros(n, dtype=np.float32)
ptx = pathlib.Path('examples/vector_add/kernel.ptx').read_text()
res = gpusim.run(ptx_src=ptx, grid=(8,1,1), block=(128,1,1),
                 params={'A':a,'B':b,'C':c,'N':n}, mode='timing', config=cfg)
res.html_report('report.html')
res.perfetto('trace.json')
print(res.summary())
# Phase 5: cluster-level metrics
print(res.cluster_summary())
# Phase 4: device-level metrics
print(res.device_summary())
print(res.cta_dispatch_events_df)
# Phase 3: Tensor Core metrics (available when kernel uses mma/wgmma)
# print(res.tc_summary())
# print(res.tc_metrics)           # dict with per-precision MMA counts
# res.mma_events_df()             # DataFrame of sync-mma events
# res.wgmma_events_df()           # DataFrame of async wgmma events
# res.tma_events_df()             # DataFrame of TMA bulk-copy events
# res.mbarrier_events_df()        # DataFrame of mbarrier phase transitions
"

# Option 2: CLI (after staging .npy inputs)
python -c "import numpy as np; np.save('a.npy', np.random.randn(1024).astype(np.float32)); np.save('b.npy', np.random.randn(1024).astype(np.float32)); np.save('c.npy', np.zeros(1024, dtype=np.float32))"
gpusim run examples/vector_add/kernel.ptx \
    --grid 8 --block 128 \
    --inputs A:a.npy,B:b.npy,C:c.npy,N:1024 \
    --mode timing \
    --output report.html --perfetto trace.json
```
- `report.html` — open in any browser; Phase 3 adds §11 Tensor Core、§12 wgmma、§13 TMA、§14 Barrier sections; Phase 4 adds §15–§18 (CTA dispatch、L2 MSHR、bulk-store overlap、per-SM utilization); Phase 5 adds §19–§20 (cluster dispatch、cluster barrier stats)
- `trace.json` — drag into https://ui.perfetto.dev; Phase 3 adds TC / TMA / Barrier tracks; Phase 4 adds per-SM swimlane; Phase 5 adds cluster swimlane

## What's modeled

### Phase status

| Phase | Status | Highlights |
|-------|--------|-----------|
| 1 | ✅ done | SIMT core, bank conflicts, coalescing, multi-CTA occupancy |
| 2 | ✅ done | Tag-precise L1/L2/HBM cache hierarchy |
| 3 | ✅ done | Tensor Core (sync mma + wgmma) + TMA-lite |
| 4 | ✅ done | Multi-SM topology, CTA scheduler, shared L2, TMA store |
| 5 | ✅ done | Hopper Cluster CGA, dsmem, cluster barrier, cluster TMA |
| 6+ | future | Warp shuffle, ITS, multi-GPU |

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

## What's NOT modeled
Warp shuffle, ITS, multi-GPU. See `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` section 11.

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
