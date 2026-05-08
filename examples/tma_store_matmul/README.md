# tma_store_matmul

Phase 4 production-pattern matmul: gmem→smem load (manual), wgmma compute,
smem→smem D writeback, TMA store smem→gmem.

## Problem

M=64, N=128, K=16 fp16 matmul with fp32 accumulation.

## Pipeline

1. All 128 threads cooperate to load A (64x16 fp16 = 2048 B) + B (16x128 fp16 = 4096 B) from gmem to smem (manual ld.global / st.shared).
2. `bar.sync` to ensure smem is fully populated.
3. `wgmma.fence` + `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16` + `commit_group` + `wait_group 0`.
4. Each thread writes its 64 fp32 D registers to smem_D (offset 6144, 32768 B = 64x128 fp32 row-major) via `st.shared.f32`.
5. `bar.sync` to ensure all D values are in smem.
6. Thread 0 only: `gpusim.tma_desc` + `cp.async.bulk.tensor.2d.global.shared::cta` + `commit_group` + `wait_group 0`.

## Smem Layout

| Region | Offset | Size |
|--------|--------|------|
| smem_A | 0      | 2048 B (64x16 fp16) |
| smem_B | 2048   | 4096 B (16x128 fp16) |
| smem_D | 6144   | 32768 B (64x128 fp32) |

## D Register Layout (wgmma spec §4.2)

For warp `w` (0..3), lane `i` (0..31), register `j` (0..63):
```
D[w*16 + i/2][(i%2)*64 + j] = %d_j
```

## Run

```
python examples/tma_store_matmul/run.py
```

## Tutorial

docs/tutorial/18-tma-store-pipeline.md
