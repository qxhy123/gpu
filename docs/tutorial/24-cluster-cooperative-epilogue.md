# Chapter 24 — Cluster Cooperative Epilogue

## TMA Store in the Cluster Direction: smem → gmem via CTA 0

Chapter 21 showed cluster TMA loads: CTA 0's thread 0 issues a single `cp.async.bulk` that writes data into shared memory — optionally into another CTA's smem using a `shared::cluster` destination pointer. The epilogue (output write-back) direction inverts this: instead of reading from HBM and writing into smem, a **cluster TMA store** reads from one or more CTAs' shared memories and writes to global memory.

The key insight of the cooperative epilogue pattern is that CTA 0 alone issues all the TMA stores — even for data that lives in CTAs 1, 2, and 3's shared memories. It encodes the source address as a **cluster-scoped smem pointer**: the high 8 bits of the smem address encode the target CTA's cluster rank, and the TMA hardware decodes that rank to read from the correct CTA's smem. CTA 0's thread does not copy the data through registers; the TMA engine performs a direct smem-to-HBM DMA for each CTA in the cluster.

This closes the deferred story from Phase 5's `cluster_matmul_dsmem`: after wgmma completes and the C-tile is distributed across cluster CTAs' smem, CTA 0 can collect all tiles and write them to global memory with four TMA store instructions — no inter-CTA data shuffling through registers needed.

The cluster-scoped smem pointer is constructed by bit-packing the rank:

```ptx
// Cluster pointer for rank r at smem offset 0:
//   cluster_ptr = (r << 24) | smem_byte_offset
// Rank 1, offset 0: 1 << 24 = 16777216
mov.u64 %rd3, 16777216;    // cluster pointer for CTA 1's smem
```

The TMA hardware on Hopper recognizes the upper byte as a rank selector and routes the read to the appropriate SM's smem port within the GPC.

## 走通 cluster_cooperative_epilogue

```bash
python examples/cluster_cooperative_epilogue/run.py
```

The kernel uses `cluster_size=4`, `grid=(4,1,1)`, `block=(32,1,1)`:

1. Every CTA fills its smem with `smem_D[tid] = rank * 1000 + tid` (32 elements per CTA).
2. All CTAs synchronize with `bar.sync 0` then `barrier.cluster.arrive` / `barrier.cluster.wait` to ensure smem is fully written before CTA 0 reads it.
3. CTA 0, thread 0 only: issues 4 TMA store instructions, one per cluster rank, each reading 128 bytes (32 × uint32) from a cluster-encoded smem address and writing to `OUT + rank * 128`.

Expected output:

```
cluster_cooperative_epilogue: cycles=<N>
  out[0:4] = [0, 1, 2, 3], out[32:36] = [1000, 1001, 1002, 1003]
```

The four TMA stores are issued consecutively by thread 0 of CTA 0, then a `cp.async.bulk.commit_group` / `wait_group 0` pair drains them all before the epilogue cluster barrier.

## 看模拟器

```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default

cfg = load_default(); cfg.cluster_size = 4; cfg.n_sm = 4
out = np.zeros(128, dtype=np.uint32)
ptx = pathlib.Path("examples/cluster_cooperative_epilogue/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
    params={"OUT": out}, mode="timing", config=cfg,
)
print("cycles:", res.metrics["cycles"])
print("hbm_write_bytes:", res.metrics.get("hbm_write_bytes", "n/a"))
```

Open `report.html` and go to **§19 Cluster TMA timeline**. The **BulkStore** panel shows the four TMA store events on CTA 0's SM. If the hardware can pipeline them (the simulator models each as an independent DMA transaction), you will see the four store intervals overlap in time, completing in roughly the same wall time as a single store of the same total data volume.

Notice that CTAs 1–3 show no BulkStore activity — they are idle at the `barrier.cluster.wait` while CTA 0 drains the stores. This idle time is the cost of the cooperative pattern: three SMs spin on a barrier while one does the output work.

## 改一改

**Independent TMA store per CTA.** Remove the `setp` guards so that every CTA issues its own TMA store for only its own smem (32 elements). Each CTA writes to `OUT + rank * 128` independently. No cluster pointer encoding needed — each CTA uses a plain smem offset of 0.

Compare the results:

- `hbm_write_bytes` stays the same: 512 bytes total (128 × uint32) regardless of who issues the stores.
- Cycle count may decrease slightly because all four SMs are active in parallel rather than three spinning on a barrier.
- However, in real production kernels with larger tiles, the cooperative pattern pays off because CTA 0 can pipeline the four TMA stores with the next iteration of the main loop, overlapping output writes with the next input fetch. Independent stores cannot do this without a more complex synchronization protocol.

The trade-off is not about raw bandwidth — it is about **latency hiding**: the cooperative epilogue lets the rest of the cluster (CTAs 1–3) continue their next main-loop work while CTA 0 manages the output.

## 真机对照

The cooperative epilogue is the standard output pattern for **CUTLASS Hopper persistent GEMM** kernels (`sm90_gemm_tma_warpspecialized_cooperative`). In that kernel:

- After each tile's wgmma completes, the C-tile is distributed across cluster CTAs' smem accumulators.
- CTA 0's epilogue warp-group issues TMA stores for all cluster CTAs using the same cluster pointer encoding shown above.
- While CTA 0 manages output, the producer warp-group in CTA 0 simultaneously fetches the next A/B tiles — hiding both store latency and HBM read latency in the same pipeline stage.

NVIDIA's documentation refers to this as the **warp-specialized cooperative epilogue**. The cluster pointer encoding (`rank << 24 | smem_offset`) is a Hopper-specific hardware feature that makes the entire pattern possible without inter-CTA register traffic. The simulator's `cluster_cooperative_epilogue` example isolates exactly this encoding and TMA store dispatch step.
