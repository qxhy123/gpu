# Chapter 15 — wgmma + TMA Pipeline

## Hopper wgmma：async + warp-group 协同

Chapters 12–14 used `mma.sync` — a *synchronous* Tensor Core instruction where all 32 lanes of a single warp collaborate. The H100's Hopper architecture introduces `wgmma.mma_async` (warp-group matrix multiply-accumulate), which operates at a larger granularity:

- **Warp group**: 4 warps = 128 threads cooperate on a single matrix tile.
- **Tile size**: `m64n128k16` for FP16 — 64 rows, 128 columns, 16 K-depth — computed by 128 threads together.
- **Async**: the instruction issues asynchronously. The warp group does not stall waiting for the Tensor Core result; it can issue further instructions while the computation is in-flight.

The wgmma instruction format in PTX:

```ptx
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {D-regs},   // 64 f32 registers per lane (64 * 128 = 8192 total outputs / 128 threads)
    smem_A,     // shared memory descriptor (smem byte offset)
    smem_B;     // shared memory descriptor
```

Key differences from `mma.sync`:
1. **Inputs from shared memory, not registers** — A and B must be in shared memory (smem), described by a 64-bit smem offset. There are no per-lane A/B register inputs.
2. **128-thread block** — one warp group = 4 warps = 128 threads. Each lane holds 64 D registers, covering `64/2 × 128/64 = 32 × 2` output tiles per lane.
3. **Explicit async lifecycle** — the instruction flow is:
   ```ptx
   wgmma.fence.sync.aligned;           // memory fence before issuing
   wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {...}, %smem_A, %smem_B;
   wgmma.commit_group.sync.aligned;    // commit the async group
   wgmma.wait_group.sync.aligned 0;    // wait for all in-flight groups to finish
   ```

## TMA-lite：dedicated copy engine + mbarrier 同步

The second major Hopper innovation is **TMA** (Tensor Memory Accelerator), a hardware copy engine that transfers tensor tiles from global memory to shared memory without occupying compute threads.

Instead of `ld.global` + `st.shared` pairs (which tie up threads and consume load/store units), TMA uses:

```ptx
// Single-thread issues the TMA copy:
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_dst], [gmem_src], [mbar_addr], {rows, cols};
```

The simulator's TMA-lite uses a simplified `cp.async.bulk` instruction that:
1. Issues an asynchronous bulk copy (the copy engine handles data movement).
2. Signals an `mbarrier` when the copy is complete.
3. The waiting threads use `mbarrier.try_wait` to poll until the data is ready.

This decouples memory transfer from computation: threads that issued the TMA copy can immediately proceed to other work (or issue the next iteration's TMA copy), while the hardware handles the actual data movement.

## examples/wgmma_basic：最小可运行 wgmma

The `examples/wgmma_basic/` kernel is the simplest possible wgmma demonstration:

1. **Copy A and B to smem** using regular `ld.global` + `st.shared` pairs (128 threads, each copying 8 fp16 elements from A and 16 from B).
2. Issue `bar.sync 0` to synchronize all 128 threads after the copies.
3. Execute one `wgmma.mma_async` for the full m64n128k16 tile.
4. Commit and wait.
5. Store the 64 D registers per lane back to global memory.

Run it:

```bash
python examples/wgmma_basic/run.py
```

Expected output:

```
wgmma_basic: A=(64, 16) B=(16, 128) -> D=(64, 128)
  max |diff| = 0.006123   PASS
```

The output shape is 64×128. Max diff ~0.006 is consistent with FP16 arithmetic rounding (K=16 accumulation, FP32 output).

The key PTX sequence in `examples/wgmma_basic/kernel.ptx`:

```ptx
bar.sync 0;                    // wait for smem copies to finish

wgmma.fence.sync.aligned;     // fence before async issue

wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {%d0, ..., %d63},          // 64 f32 regs per lane
    %rd9,                      // smem_A offset = 0
    %rd10;                     // smem_B offset = 2048

wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

## examples/wgmma_async_pipeline：K-tile 流水线

The `examples/wgmma_async_pipeline/` example adds a K-tile loop: it computes an M=64, N=128, K=256 matmul by iterating over 16 K-tiles of size k=16. Each iteration:

1. **TMA copy**: issue `cp.async.bulk` to load the next A-tile (64×16 fp16) and B-tile (16×128 fp16) into shared memory.
2. **Wait on mbarrier**: poll `mbarrier.try_wait` until both tiles are in smem.
3. **wgmma**: execute the async matrix multiply on the loaded tiles.
4. **Commit + wait**: ensure the wgmma result is ready before the next iteration.

```bash
python examples/wgmma_async_pipeline/run.py
```

Expected output:

```
wgmma_async_pipeline: max diff=3.91e-03
```

The low error confirms correct accumulation across 16 K-tiles. The pipeline structure is single-buffered here (one smem buffer, one mbarrier) — the TMA copy and wgmma are serialized within each iteration.

## 看模拟器：HTML 报告中的 in-flight 时段

Run in timing mode to generate a report:

```python
import numpy as np, pathlib, gpusim

A = np.random.randn(64, 16).astype(np.float16)
B = np.random.randn(16, 128).astype(np.float16)
out = np.zeros(64 * 128, dtype=np.float32)

ptx = pathlib.Path("examples/wgmma_basic/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(1,1,1), block=(128,1,1),
    params={"A": A.flatten(), "B": B.flatten(), "OUT": out},
    mode="timing",
)
print("cycles:", res.metrics.get("cycles"))
```

Open `report.html`. In the Perfetto trace:
- Look for `wgmma.mma_async` events across all 4 warps of the warp group. They issue simultaneously (warp-group sync).
- After `wgmma.commit_group`, the warps may issue other instructions while the Tensor Core is busy.
- The `wgmma.wait_group` stall shows as a `WGMMA_WAIT` stall type — this is the cycle budget where threads are blocked waiting for the async result.

For `wgmma_async_pipeline` with K_TILES=16, look for the repeating pattern:
```
TMA issue → mbarrier wait (MEMORY stall) → wgmma async → wait (WGMMA_WAIT stall) → next tile
```

In a double-buffered pipeline, the TMA for tile k+1 would overlap with the wgmma for tile k. The single-buffered version here shows no overlap — the `mbarrier.try_wait` must complete before `wgmma.mma_async` issues.

## 改一改

**Single-buffer vs. overlap:** In a double-buffered pipeline, two smem regions ping-pong:
- Buffer A: holds tile k, used by current wgmma
- Buffer B: being filled by TMA for tile k+1

The `wgmma_async_pipeline` kernel is single-buffered. To see the overhead:

1. Measure the cycle count with K_TILES=16.
2. Conceptually, a double-buffered version could overlap TMA latency with wgmma compute. The ratio: `(TMA latency) / (wgmma latency)` determines how much overlap is possible.

Modify `run.py` to try different K_TILES values:

```python
for k_tiles in [1, 4, 8, 16]:
    res = gpusim.run(
        ptx_src=ptx, grid=(1,1,1), block=(128,1,1),
        params={"A": A.flatten(), "B": B.flatten(), "OUT": out, "K_TILES": k_tiles},
        mode="timing",
    )
    print(f"K_TILES={k_tiles}: cycles={res.metrics.get('cycles')}")
```

If cycles scale linearly with K_TILES, there is no overlap (single-buffered, serial). A double-buffered pipeline would show near-constant overhead for small K_TILES and better cycles/tile for large K_TILES.

**Disable async:** Change the mbarrier expected-count from 2 to 0 (or bypass TMA altogether and use regular `ld.global`). This forces all copies through the thread-based path. Cycle count should increase as the wgmma is forced to wait for thread copies to finish.

## 真机对照

CUTLASS Hopper kernels (from the `cutlass/examples/48_*` examples and `cutlass/include/cute/`) universally use:
1. **TMA** for all global→shared transfers (both A and B tiles)
2. **wgmma.mma_async** for all compute
3. **Double or triple buffering** — two or three smem buffer sets so TMA overlap with wgmma is maximized

The H100 datasheet notes:
- TMA supports up to 128 concurrent bulk transfers
- wgmma supports m64n{64,128,256}k16 for FP16/BF16, m64n{64,...,256}k32 for FP8
- The warp group of 128 threads is a fundamental hardware unit for wgmma (4 warps must always cooperate)

Real production kernels like Flash Attention v3 (Tri Dao et al., 2024) use TMA + wgmma with double-buffered pipelines to achieve near-peak utilization of the H100 Tensor Cores. The simulator captures the instruction structure and async lifecycle, making it possible to reason about pipeline behavior before touching real hardware.
