# Chapter 07 — Tiled Matrix Multiplication

## Putting it together

The previous chapters covered the individual building blocks: SIMT execution, warp scheduling, coalescing, bank conflicts, divergence, and occupancy. Tiled matrix multiplication combines all of these in one kernel.

The `examples/tiled_matmul/` example computes `C = A × B` for 16×16 float32 matrices using a single CTA. It demonstrates:
- **Coalesced global loads** — all 256 threads load consecutive elements from A and B.
- **Shared memory tile staging** — tiles are read into `smem_A` and `smem_B` before the compute loop.
- **`bar.sync` synchronization** — ensures all threads have written their tile element before any thread starts reading.
- **k-loop with smem access** — the dot product loop reads two smem arrays without bank conflicts.

## Dataflow diagram

```
Global memory:         Shared memory:         Registers:
 A[16×16]              smem_A[16×16]          acc (per thread)
  │                     │                      │
  │  ld.global (tile)   │  ld.shared (k-loop)  │
  └─────────────────────┘  ────────────────────┤
 B[16×16]              smem_B[16×16]           │ fma.f32
  └─────────────────────┘  ────────────────────┘
                                               │
                                   st.global → C[16×16]
```

Each of the 256 threads (tid.x = col, tid.y = row) loads one element of A and one element of B into shared memory. After `bar.sync`, every thread independently computes `sum_{k=0}^{15} smem_A[row*16+k] * smem_B[k*16+col]` using 16 FMA instructions. The result is stored to C.

## Walking the simulator output

Run the example:

```bash
python examples/tiled_matmul/run.py
```

Output:

```
max abs error: 0.0
```

The result is numerically exact (within float32 precision). Timing metrics:

```
cycles: 1942
occupancy: {'active_ctas': 8, 'bottleneck': 'warps'}
```

**Global loads (coalescing):** The 256-thread block has 8 warps. Warp 0 contains threads (0,0)–(15,1) — threads with col 0–15, row 0 for 16 threads, then col 0–15 row 1 for 16 threads. Each warp loads 32 consecutive float32 elements from A (and 32 from B). These are perfectly coalesced — 1 transaction per warp.

**`bar.sync` cost:** After the global loads, all 8 warps hit `bar.sync`. In timing mode, this means each warp waits for all 7 others to also arrive. The `BARRIER` stall count in the report reflects this. With 8 warps and approximately equal progress, the barrier overhead is minimal — each warp arrives at nearly the same cycle.

**k-loop smem access:** In each of the 16 k-iterations, every thread reads `smem_A[row*16+k]` and `smem_B[k*16+col]`.

For `smem_A`: all threads with the same row read `row*16+k` — threads in different rows read different addresses at the same k, so no two active-at-the-same-cycle threads share a bank.

For `smem_B`: thread `(col, row)` reads `(k*16+col)*4+1024`. For fixed k, threads with different columns read addresses `k*16*4 + col*4 + 1024`. The 32 consecutive col values (0–15, 0–15 across two rows) map to banks `0–15, 0–15` — a 2-way conflict. However, within a single warp (which has 32 threads covering col 0–15 for two different rows), the bank access pattern is `(col + 0) % 32` and `(col + 0) % 32` — actually no conflict since only one thread per column in a half-warp accesses each bank.

Open `report.html` and check the **bank conflict histogram** for the two `ld.shared` instructions in the k-loop. Both should show conflict degree 1.

**FMA throughput:** After both smem reads, each lane issues `fma.f32`. The FMA latency is 4 cycles, throughput 1/cycle. With 8 warps and a 4-cycle FMA, the scheduler can pipeline FMAs from different warps to fill the issue slots. The total cycle count of ~1942 for 16 k-iterations × 3 instructions (2 ld.shared + 1 fma) × 8 warps = ~384 compute cycles plus ~400 cycles for the initial global loads plus barrier and store overhead.

## What's missing for "real" matmul

This example computes a 16×16 matrix product in a single tile. Real matrix multiplication requires:

1. **K-loop over multiple tiles** — for M×N×K with K > 16, the outer k-loop iterates over tiles, accumulating into registers. Each tile iteration requires a new global load + `bar.sync`.

2. **Register accumulation across tiles** — the FP32 accumulator `%f3` is already in a register, so inter-tile accumulation just skips re-initializing it.

3. **FP16/BF16 operands + Tensor Core** — real production kernels use `wmma` instructions (or PTX-level `mma.sync`) to perform 16×16×16 matrix multiply-accumulate in a single instruction. Tensor Core throughput is ~16× higher than scalar FMA for the same data. This requires Phase 3 support.

4. **Swizzled/transposed shared memory layouts** — to avoid bank conflicts when the tile's access pattern hits power-of-2 strides, production kernels use padding or swizzling. Chapter 04 covers the principle.

5. **Double-buffering** — overlap global loads for tile k+1 with computation on tile k using `cp.async` and two ping-pong smem buffers. This requires Phase 2 async copy support.

## 改一改 — Block size (16,16) → (32,8)

Change the test to `block=(32,8,1)`. Now there are still 256 threads but they are arranged as 32 columns × 8 rows. The matrix is still 16×16, so the thread-to-matrix mapping no longer covers the full matrix directly (32 threads in x but only 16 columns).

To make this work, you would need to guard against `col >= 16` — threads with col 16–31 should not load or store. This demonstrates a key occupancy/workload trade-off: a wider block might give better coalescing for the global load (32 threads load 32 consecutive elements in one transaction) but introduces boundary checking divergence.

With `block=(32,8,1)` and 32 warps per SM cap, the occupancy is: `32 warps / (256/32 = 8 warps per CTA) = 4 CTAs`. Compare this to the original `block=(16,16,1)` with `8 warps per CTA` → also 4 CTAs (limited by warp cap). The occupancy is the same; the difference is in the access pattern and the boundary-check divergence.

## 真机对照

Skipped — no reference fixtures committed. On a real H100, tiled matmul with a 16×16 single tile would run in a few hundred nanoseconds. The Tensor Core path would be ~16× faster. The simulator does not model Tensor Core, so the cycle counts reflect only the scalar FMA path — useful for understanding the baseline before Tensor Core optimization.
