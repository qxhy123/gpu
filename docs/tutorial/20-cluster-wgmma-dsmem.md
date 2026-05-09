# Chapter 20 — Cluster + wgmma + dsmem

## 协同模式：一 CTA 拉数据，全 Cluster 共用

Chapter 15 showed how a single CTA uses TMA to load tiles and wgmma to compute. Within that model every CTA independently fetches its own A and B tiles from HBM — if four adjacent CTAs in a cluster all need the same A tile (a column-parallel tiling strategy), that tile is fetched four times.

The **dsmem** (distributed shared memory, i.e. `shared::cluster` access) pattern breaks this redundancy. One designated CTA — say rank 0 — loads the shared tile into its own smem. The other CTAs then read it directly via `mapa.shared::cluster` + `ld.shared::cluster`, traversing only the intra-GPC interconnect. HBM sees one fetch instead of four. This is the foundation of Hopper's cluster-cooperative matmul strategy.

## mapa.shared::cluster：本地指针 → 远程指针

The `mapa` instruction converts a local smem byte offset into a cluster-scoped pointer that names a specific CTA's smem:

```ptx
// All CTAs want to read CTA 0's smem starting at byte offset 0.
mov.u64 %rd_local, 0;         // smem byte offset in the *target* CTA
mov.u32 %r_target_rank, 0;    // target rank = CTA 0
mapa.shared::cluster %rd_remote, %rd_local, %r_target_rank;
// %rd_remote encodes (rank=0 << 24) | offset=0
```

The encoded address is then used with `ld.shared::cluster` to fetch data from rank 0's smem into the requesting CTA's registers:

```ptx
ld.shared::cluster.f32 %f0, [%rd_remote];
```

Every CTA in the cluster can issue this independently and concurrently; the hardware routes each request over the GPC interconnect to the owning SM's smem banks.

## 数据共享 vs 独立 smem 的 Trade-off

Using dsmem for shared tiles introduces a structural asymmetry: CTA 0 bears the cost of the global memory fetch (TMA or thread-based), while all CTAs share the smem capacity of CTA 0's buffer. The trade-offs are:

| Strategy | HBM traffic | Smem pressure | Interconnect use |
|---|---|---|---|
| Each CTA fetches independently | 4× (one per CTA) | Local only | None |
| CTA 0 fetches, others use dsmem | 1× | CTA 0 smem shared | Intra-GPC |

For a 64×16 FP16 A-tile (2 KB), four independent fetches cost 8 KB of HBM bandwidth per cluster; the dsmem version costs 2 KB. The saving grows with cluster size and tile reuse factor. The cost is that CTA 0's smem must remain stable (not reused) for the duration of the dsmem reads — a lifetime-management requirement that must be respected in pipelined kernels.

## 走通 cluster_matmul_dsmem

The current `examples/cluster_matmul_dsmem/kernel.ptx` is a **simplified** version that demonstrates the dsmem mechanism without wgmma. CTA 0 loads 128 FP32 elements from `A` into its smem. After a cluster barrier, all four CTAs read their 32-element slice from CTA 0's smem via `mapa` + `ld.shared::cluster` and write the results to `OUT`.

```bash
python examples/cluster_matmul_dsmem/run.py
```

Expected output:

```
cluster_matmul_dsmem (simplified): cycles=680, max diff=0.00e+00
```

Zero diff confirms that dsmem reads correctly deliver CTA 0's data to all cluster members. The kernel uses a 4-CTA cluster (`cfg.cluster_size = 4`) with a 128-thread block in each CTA.

Note: a full `wgmma + dsmem` version (where CTA 0 fetches the A-tile and all CTAs feed it into `wgmma.mma_async`) requires the simulator to support the `m64n32k16` tile shape used in narrow-N configurations. That variant is deferred to Phase 6+; the current example isolates the dsmem addressing and synchronisation mechanics that underpin the full design.

## 看模拟器

Run in timing mode and inspect the `dsmem_remote_rate` metric:

```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default

cfg = load_default()
cfg.cluster_size = 4; cfg.n_sm = 4
rng = np.random.RandomState(0)
A   = (rng.rand(128) * 100).astype(np.float32)
out = np.zeros(128, dtype=np.float32)

ptx = pathlib.Path("examples/cluster_matmul_dsmem/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(4,1,1), block=(128,1,1),
    params={"A": A.copy(), "B": np.zeros(1, dtype=np.float32), "OUT": out},
    mode="timing", config=cfg,
)
print("cycles:", res.metrics["cycles"])
print("dsmem_remote_rate:", res.metrics.get("dsmem_remote_rate", "n/a"))
```

`dsmem_remote_rate` reports the fraction of `ld.shared` accesses that resolved against a *remote* CTA's smem (as opposed to the local CTA's). In this kernel, CTAs 1–3 perform all their element reads via dsmem, so the rate should be close to 0.75 (3 out of 4 CTAs are purely remote readers).

Open `report.html` at **§19**. The cluster barrier timeline shows CTA 0 completing its global load before `arrive`, while CTAs 1–3 skip straight to the barrier — their workload is front-loaded on CTA 0.

## 改一改

**Replace mapa with local ld.shared:** Modify the kernel so each CTA independently loads its slice from global memory (remove the `mapa` + `ld.shared::cluster` path and add a direct `ld.global` for each CTA). Rerun and compare:

- `dsmem_remote_rate` drops to 0 (no remote smem accesses).
- Total HBM reads increase (visible in `res.metrics["hbm_read_bytes"]` or the memory timeline in `report.html`): each CTA issues its own 32-element fetch, totalling 4× the original HBM traffic.
- Cycle count may decrease (no `mapa`/interconnect overhead) or stay similar — the win from dsmem is HBM bandwidth, not latency, and becomes decisive only when HBM is the bottleneck.

This experiment confirms the dsmem value proposition: you trade intra-GPC interconnect usage for HBM bandwidth reduction.

## 真机对照

The production example of this pattern is **CUTLASS Hopper persistent matmul** (`cutlass/examples/57_hopper_grouped_gemm` and the `sm90_gemm_tma_warpspecialized_cluster` kernel family). In those kernels, the cluster is used to share the B-tile (or A-tile for column-split tiling): one CTA per cluster issues a TMA load into its smem, and the wgmma instructions in every CTA use `mapa`-encoded pointers so the Tensor Cores consume data from the owning CTA's smem without redundant HBM fetches. Measured on H100, a 4-CTA cluster achieves ~3.8× HBM bandwidth reduction on the shared tile, enabling the matmul to remain compute-bound at larger batch sizes than a non-cluster kernel would.
