# wgmma_async_pipeline

TMA + wgmma pipelined matrix multiply: M=64, N=128, K=256 decomposed into 16
K-tiles of size 16. Each tile is loaded from global memory via TMA (Tensor Memory
Accelerator) and consumed by wgmma asynchronous tensor core ops.

## Pattern

For each K-tile iteration:
1. Re-init mbarrier (expected=2 arrivals — one per TMA transfer)
2. Issue `gpusim.tma_desc` + `cp.async.bulk.tensor.2d` for A-tile (64x16 fp16)
3. Issue `gpusim.tma_desc` + `cp.async.bulk.tensor.2d` for B-tile (16x128 fp16)
4. Spin on `mbarrier.try_wait` until both TMA copies complete (phase flip from 0 to 1)
5. `wgmma.mma_async` reads the smem tile and accumulates into D registers
6. `wgmma.commit_group` + `wgmma.wait_group 0`

After 16 iterations, write the 64x128 f32 accumulator back to OUT.

## Usage

```bash
python examples/wgmma_async_pipeline/run.py
```

## Testing

```bash
.venv/bin/pytest tests/parity/test_wgmma_async_pipeline.py -v
```

## Smem layout

| Region   | Offset | Size  | Content                    |
|----------|--------|-------|----------------------------|
| smem_A   | 0      | 2048  | 64x16 fp16 A-tile          |
| smem_B   | 2048   | 4096  | 16x128 fp16 B-tile         |
| mbar0    | 6144   | 8     | mbarrier state             |
