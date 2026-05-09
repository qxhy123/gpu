# Chapter 21 — Cluster TMA + mbarrier Pipeline

## Cluster TMA：CTA 0 代理 Fetch，TMA 写入目标 CTA 的 smem

Chapter 20 used thread-based loads (CTA 0 runs `ld.global` in a loop) to fill its smem before the other CTAs read it over dsmem. The production pattern goes one step further: CTA 0's thread 0 issues a **TMA load** (a single `cp.async.bulk` instruction), and the TMA hardware engine writes the fetched tile directly into smem — including, via the `shared::cluster` destination address, into *another CTA's* smem. This eliminates thread occupancy during the fetch and enables a clean producer-consumer split across the cluster.

The address passed to the TMA instruction as the smem destination can be a cluster-encoded pointer:

```ptx
// CTA 0 thread 0: TMA loads into its own smem_T at offset 0.
// The TMA engine writes there; all CTAs will read it via dsmem.
mov.u64 %rd_smem_dst, 0;     // byte offset 0 = smem_T in CTA 0
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [%rd_smem_dst], [%rd_tma_desc], [%rd_mbar];
```

The qualifier `shared::cluster` on the `cp.async.bulk` instruction tells the TMA unit to interpret the destination as a cluster-scoped address, allowing it to write into any CTA's smem within the cluster. Here we keep it simple: the destination is CTA 0's own smem (offset 0), and the other CTAs read it via `mapa` + `ld.shared::cluster` after the barrier.

## Cluster mbarrier：dsmem 上的多 CTA producer-consumer 同步

A regular `mbarrier` lives in one CTA's smem and is only visible to that CTA's threads. In a cluster pipeline we need all CTAs to know when the TMA load is complete. The solution is to create the `mbarrier` in CTA 0's smem and let the other CTAs observe it through dsmem:

```ptx
// CTA 0 thread 0 initialises the barrier with expected-arrival-count = 1.
mov.u64 %rd_mbar, 1024;              // smem offset of the mbarrier
mbarrier.init.shared::cta [%rd_mbar], 1;
```

After CTA 0 issues the TMA copy, the TMA engine will decrement the barrier's pending-count and flip its phase when the copy completes. All CTAs — including the ones whose threads did no issuing — can then spin on the mbarrier via `mapa` to detect completion:

```ptx
// Any CTA: encode CTA 0's mbarrier address
mov.u64 %rd_mbar_off, 1024;
mov.u32 %r_cta0, 0;
mapa.shared::cluster %rd_mbar_remote, %rd_mbar_off, %r_cta0;
// ... poll mbarrier.try_wait on %rd_mbar_remote ...
```

In the current `cluster_tma_pipeline` example, this two-phase synchronisation is simplified to a single `barrier.cluster.arrive` / `barrier.cluster.wait` pair: CTA 0 arrives after issuing the TMA copy (meaning the *issue* has happened, not the data transfer), and the cluster barrier guarantees that the TMA has resolved before anyone proceeds, because the TMA engine writes to CTA 0's smem before the barrier wait unblocks. This matches the simulator's model of TMA completion ordering with respect to `barrier.cluster`.

## 走通 cluster_tma_pipeline

```bash
python examples/cluster_tma_pipeline/run.py
```

Expected output:

```
cluster_tma_pipeline: cycles=510, max diff=0.00e+00
```

The kernel uses `cluster_size=4`, `grid=(4,1,1)`, `block=(32,1,1)`. CTA 0's thread 0:

1. Initialises an `mbarrier` at smem offset 1024 with count 1.
2. Creates a TMA descriptor for 256 FP32 elements (1024 bytes) from `SRC`.
3. Issues `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` to load all 256 elements into CTA 0's smem at offset 0.

All four CTAs then execute `barrier.cluster.arrive` / `barrier.cluster.wait`. After the barrier, each CTA uses `mapa(offset=0, rank=0)` to build a remote pointer into CTA 0's smem and reads its 64-element slice (two elements per thread, since there are only 32 threads per CTA) via `ld.shared::cluster.f32`.

## 看模拟器

Run in timing mode:

```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default

cfg = load_default(); cfg.cluster_size = 4; cfg.n_sm = 4
rng = np.random.RandomState(0)
src = (rng.rand(256) * 100).astype(np.float32)
out = np.zeros(256, dtype=np.float32)

ptx = pathlib.Path("examples/cluster_tma_pipeline/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
    params={"SRC": src.copy(), "OUT": out},
    mode="timing", config=cfg,
)
print("cycles:", res.metrics["cycles"])
print("dsmem_remote_rate:", res.metrics.get("dsmem_remote_rate", "n/a"))
```

Open `report.html`:

- **§19 Cluster TMA timeline** — shows the TMA bulk copy event on CTA 0's SM, the `CLUSTER_BARRIER_WAIT` stall on CTAs 1–3 while the TMA is in flight, and the `ld.shared::cluster` read burst from all four CTAs once the barrier resolves.
- **§20 dsmem bandwidth** — shows the intra-GPC interconnect traffic spike as CTAs 1–3 each fetch their 64-element slice from CTA 0's smem.

The `cluster_summary()` API (if available in your build) prints per-SM statistics for cluster barrier arrivals and dsmem bytes transferred:

```python
if hasattr(res, "cluster_summary"):
    print(res.cluster_summary())
```

## 改一改

**Each CTA issues its own TMA:** Remove the `if rank == 0` guard and have every CTA issue a TMA load covering only its own 64-element slice (update the TMA descriptor dimensions accordingly). Remove `mapa` and have each CTA write to its own smem, then read locally.

Compare:

- `dsmem_remote_rate` drops to 0.
- `hbm_read_bytes` increases by ~4×: four TMA engines each fetch 256 bytes instead of one fetching 1024 bytes total — same data volume, but four separate HBM transactions with their own request overhead.
- The `CLUSTER_BARRIER_WAIT` stall shrinks because each CTA no longer waits for a neighbour's TMA; the cluster barrier is now a trivial rendezvous with no data dependency.

This quantifies the bandwidth efficiency gain of the cluster-TMA pattern: one coordinated fetch costs roughly the same latency as four independent fetches but saves proportional HBM bandwidth when the fetched data is shared.

## 真机对照

The cluster TMA pipeline maps directly to the **mainloop of CUTLASS's warp-specialized Hopper GEMM** (`sm90_gemm_tma_warpspecialized`). In that kernel:

- A dedicated **producer warp-group** in CTA 0 runs the TMA load loop, issuing `cp.async.bulk` instructions into a double-buffered smem region.
- **Consumer warp-groups** across all cluster CTAs run `wgmma.mma_async`, reading the shared A-tile via `mapa`-encoded smem pointers.
- **Cluster mbarriers** (stored in CTA 0's smem, observed by all CTAs through dsmem) gate each pipeline stage: consumers wait for the producer's TMA to complete before starting wgmma, and the producer waits for consumers to finish before overwriting the buffer.

This architecture hides both TMA latency (behind wgmma compute) and HBM bandwidth pressure (via dsmem sharing), allowing Hopper matmuls to achieve >95% of peak Tensor Core throughput on large tiles. The simulator's `cluster_tma_pipeline` example captures the essential synchronization structure — barrier lifecycle, mapa encoding, and dsmem read ordering — before you tackle the full production kernel.
