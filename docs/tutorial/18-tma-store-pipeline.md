# Chapter 18 — TMA Store 与端到端生产 Matmul Pipeline

## 为什么 TMA Store 用 commit/wait_group 而不是 mbarrier

Chapter 15 introduced TMA for *loads* (global→shared), synchronized with `mbarrier`. The store direction (shared→global) uses a different synchronization primitive: `cp.async.bulk.commit_group` / `cp.async.bulk.wait_group`.

The asymmetry has a hardware reason. TMA load fills a destination smem buffer that *multiple threads will read* — so the completion signal needs to be a broadcast that all waiting threads can observe (mbarrier is designed for this: it has an arrival count, a phase bit, and a `try_wait` poll). TMA store *drains* a smem buffer that threads have already finished writing — the completion signal only needs to tell the issuing agent (usually a single thread or warp-group) that the data has left the chip and smem is safe to reuse. A group counter (`commit_group` / `wait_group N`) is sufficient and simpler.

The lifecycle of a TMA store:

```ptx
// Thread 0 issues the store after all threads have written smem_D:
bar.sync 0;                                          // wait for smem_D writes
cp.async.bulk.tensor.2d.global.shared::cta
    [tma_desc], [smem_src_offset];                  // issue async bulk store
cp.async.bulk.commit_group;                          // commit the current group
cp.async.bulk.wait_group 0;                          // wait: 0 in-flight groups remain
```

`commit_group` groups all preceding uncommitted `cp.async.bulk` stores into one trackable unit. `wait_group N` stalls until at most N groups remain in flight. `wait_group 0` means: wait until *all* committed groups have completed and data has reached global memory.

## BulkStoreQueue：每 warp-group 的异步 store 队列

The simulator models a **BulkStoreQueue** per warp-group. When `cp.async.bulk.commit_group` executes, the current batch of bulk stores is pushed onto the queue as a named group. The queue tracks:

1. **Issued cycle**: when the store command was dispatched to the copy engine.
2. **Committed cycle**: when `commit_group` was called.
3. **Completion cycle**: when the copy engine confirms the data has reached global memory.

`wait_group N` checks the queue length. If more than N groups are in flight, the warp-group stalls until enough groups drain. This is visible in the simulator's Perfetto trace as `BULK_STORE_WAIT` stall events.

## 走通 tma_store_matmul

The `examples/tma_store_matmul/` kernel is a complete end-to-end matmul:

1. **Load A and B to smem** — 128 threads each copy 8 fp16 elements from A and 16 fp16 elements from B using `ld.global` + `st.shared` pairs (same pattern as `wgmma_basic`).
2. **`bar.sync 0`** — synchronize all 128 threads to ensure smem is fully written before the warp-group starts the wgmma.
3. **wgmma** — one `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16` over the full tile. The 128 threads of the warp-group collectively compute a 64×128 FP32 output tile.
4. **`wgmma.commit_group` + `wait_group 0`** — wait for the wgmma result to land in the 64 D registers per lane.
5. **Write D registers to smem_D** — each thread stores its 64 FP32 registers into the `smem_D` buffer at offset 6144. The smem layout maps warp, lane, and register index to row-major output coordinates.
6. **`bar.sync 0`** — synchronize all threads: smem_D is now fully populated.
7. **Thread 0 issues TMA store** — a single thread creates a TMA descriptor for the output matrix, then issues `cp.async.bulk.tensor.2d.global.shared::cta` to copy the 64×128 FP32 smem_D buffer to global memory.
8. **`commit_group` + `wait_group 0`** — wait for the TMA store to complete before returning.

The PTX sequence for the TMA store (thread 0 only):

```ptx
// Create TMA descriptor for OUT (64 rows x 128 cols fp32)
gpusim.tma_desc %rd12, %rd2, 128, 64, 128, 4;

// Issue TMA store: smem_D (at offset 6144) -> OUT
mov.u64 %rd13, 6144;
cp.async.bulk.tensor.2d.global.shared::cta [%rd12], [%rd13];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

Run it:

```bash
python examples/tma_store_matmul/run.py
```

Expected output:

```
tma_store_matmul: cycles=1240, max diff=3.91e-03
```

The max diff (~0.004) reflects FP16 arithmetic rounding across the 16-element K dimension. The correctness check compares against NumPy's FP32 reference.

## 看模拟器

Generate a timing report:

```python
import numpy as np, pathlib, gpusim

A = np.random.randn(64, 16).astype(np.float16)
B = np.random.randn(16, 128).astype(np.float16)
out = np.zeros(64 * 128, dtype=np.float32)

ptx = pathlib.Path("examples/tma_store_matmul/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(1,1,1), block=(128,1,1),
    params={"A": A.flatten(), "B": B.flatten(), "OUT": out},
    mode="timing",
)
print("cycles:", res.metrics["cycles"])
print("bulk_store_overlap:", res.metrics.get("bulk_store_async_overlap", 0))
```

Open `report.html` and navigate to **§18 BulkStore timeline**. The timeline shows:

- The wgmma compute phase as a shaded block across all 4 warps.
- The `bar.sync` stall between smem_D writes and the TMA store issue.
- The TMA store as a separate copy-engine event, running after `commit_group`.
- The `BULK_STORE_WAIT` stall on thread 0 while `wait_group 0` spins.

The metric `bulk_store_async_overlap` measures the fraction of TMA store cycles where compute is also in progress. For this single-tile kernel the value is 0: the `wait_group 0` on thread 0 blocks until the store finishes, and no compute issues during that stall. In a pipelined multi-tile kernel, TMA stores for tile k can overlap with wgmma for tile k+1 — `bulk_store_async_overlap` would then be nonzero.

## 改一改

**Induce a data race:** Move `cp.async.bulk.wait_group 0` to *before* the `wgmma.wait_group` — or equivalently, issue the TMA store before `bar.sync 0` ensures smem_D is populated:

```ptx
// WRONG: TMA store issued before smem_D is fully written
cp.async.bulk.tensor.2d.global.shared::cta [%rd12], [%rd13];
cp.async.bulk.commit_group;
// ... wgmma and st.shared happen here, but TMA store already in flight ...
cp.async.bulk.wait_group 0;
```

The simulator's memory consistency checker will flag this as a `SMEM_RACE`: the TMA copy engine reads smem_D bytes at the same cycle that `st.shared.f32` instructions from compute threads are still writing them. The output will be partially or fully garbage. This demonstrates why the `bar.sync 0` before the TMA store issue is mandatory — it is not a performance concern but a correctness fence.

**Try `wait_group 1` instead of `wait_group 0`:** In a multi-tile loop, using `wait_group 1` (allow one in-flight group) lets the kernel issue the next tile's TMA store before waiting for the previous one to complete. Modify the kernel to run two tiles sequentially:

```ptx
// Tile 0: wgmma → smem_D → TMA store → commit_group
// Tile 1: wgmma → smem_D → TMA store → commit_group
// wait_group 0   <-- waits for both groups at the end
```

Compare cycle count to the version with `wait_group 0` after each tile. The pipelined version should show lower total stall time if the TMA store for tile 0 overlaps with the compute for tile 1.

## 真机対照

The `tma_store_matmul` pattern is the foundation of CUTLASS's **Hopper persistent matmul** kernel (the `sm90_gemm_tma_warpspecialized` family). Production CUTLASS kernels extend this in three ways:

1. **Warp specialization**: separate "producer" warp-groups issue TMA loads and stores, while "consumer" warp-groups run wgmma. This avoids thread 0 serialization.
2. **Double-buffered smem**: two smem_D buffers ping-pong — while the TMA store drains buffer A to global memory, compute fills buffer B. `wait_group 1` allows one in-flight store group, enabling full overlap.
3. **Epilogue fusion**: instead of storing raw FP32 D values, the output pipeline applies scale + bias + activation before the TMA store, all within smem, before the bulk copy to global memory.

The H100 TMA unit supports up to 128 concurrent in-flight bulk transfers, and the copy engine runs entirely off the critical path of the SM's warp scheduler. Measured on real hardware, a well-pipelined Hopper matmul achieves >95% of the theoretical wgmma peak because TMA store latency is almost completely hidden behind compute — exactly the behavior the simulator lets you reason about before writing a single line of CUDA.
