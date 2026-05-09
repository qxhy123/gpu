# cluster_matmul_dsmem

Phase 5 cluster + dsmem demo. 4-CTA cluster: CTA 0 loads 128 fp32 values from
gmem into its smem; all 4 CTAs use `mapa.shared::cluster` to map CTA 0's smem
into a cross-cluster address; each CTA reads its 32-element slice via
`ld.shared::cluster.f32` and writes to OUT.

This is a simplified version focused on the **dsmem mechanism** (no wgmma).
The educational value is demonstrating cluster-shared memory access patterns
using Hopper's distributed shared memory (dsmem) instructions.

## Run

```
python examples/cluster_matmul_dsmem/run.py
```

## Kernel design

- Grid: (4, 1, 1) with cluster_size=4 — one cluster of 4 CTAs
- Block: (128, 1, 1) — 4 warps per CTA
- CTA rank 0: all 128 threads load A[0..127] into smem[0..511]
- `bar.sync 0` (intra-CTA) then `barrier.cluster.{arrive,wait}` (inter-CTA)
- All CTAs: threads 0..31 use `mapa.shared::cluster` to get CTA 0's smem base
- `ld.shared::cluster.f32` reads element at rank*32+tid from CTA 0's smem
- Writes to OUT[rank*32+tid]

## Tutorial

docs/tutorial/20-cluster-wgmma-dsmem.md
