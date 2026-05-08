# Chapter 09 — Shared Memory vs. L1 Cache

## Two kinds of fast memory

Both shared memory (smem) and the L1 data cache are physical SRAM on the SM chip. In hardware they share the same 256 KB SRAM bank — you configure the split at compile time. In the simulator they are independent objects, but they compete for the same SIMT "fast path":

| Property | Shared memory | L1 cache |
|----------|--------------|---------|
| Management | Manual (you write indices) | Automatic (hardware LRU) |
| Scope | Per-CTA | Per-SM (shared across CTAs) |
| Hit latency | ~1 cycle (+ bank-conflict penalty) | 25 cycles |
| Access pattern | Programmer-controlled | LRU decides |
| Capacity | Up to 228 KB / SM (configurable) | 128 KB (default) |
| Reuse guarantee | Yes — you control what goes in | No — LRU may evict |

Chapter 04 covered shared memory bank conflicts. Chapter 08 introduced the L1 cache. This chapter compares the two using the `smem_vs_l1_demo` example.

## The experiment — same computation, two strategies

`examples/smem_vs_l1_demo/` contains two PTX variants of a 16×16 float32 matrix multiply:

- **kernel_smem.ptx** — tiles A and B into shared memory first, then loops through k in SRAM. This is the Phase 1 tiled_matmul kernel.
- **kernel_no_smem.ptx** — loops through k reading directly from global memory each iteration, relying on the L1 cache to retain recently used elements.

Run both:

```bash
python examples/smem_vs_l1_demo/run.py
```

Output:

```
kernel_smem.ptx: cycles=1212, max_err=0.000000
kernel_no_smem.ptx: cycles=1298, max_err=0.000000
```

Both produce numerically correct results. The smem version is **86 cycles (7%) faster**.

## What the cache metrics reveal

```python
import numpy as np, pathlib, gpusim

rng = np.random.RandomState(0)
A = rng.randn(16, 16).astype(np.float32)
B = rng.randn(16, 16).astype(np.float32)
here = pathlib.Path("examples/smem_vs_l1_demo")

for variant in ("kernel_smem.ptx", "kernel_no_smem.ptx"):
    C = np.zeros((16, 16), dtype=np.float32)
    ptx = (here / variant).read_text()
    res = gpusim.run(ptx_src=ptx, grid=(1, 1, 1), block=(16, 16, 1),
                     params={"A": A, "B": B, "C": C}, mode="timing")
    print(f"{variant}: L1 hit={res.cache_metrics['l1_hit_rate']:.2%}, "
          f"HBM events={len(res.hbm_events_df)}")
```

Output:

```
kernel_smem.ptx: L1 hit=0.00%, HBM events=24
kernel_no_smem.ptx: L1 hit=71.88%, HBM events=24
```

This is the key result. Both variants issue the same number of HBM accesses (24 events) — because the total data to fetch from global memory is the same: 16×16 A + 16×16 B = 512 floats = 4 cache lines per tile. The difference is in the L1:

- **smem version**: 0% L1 hit rate. The smem version does `ld.global` once into smem (16 lines of global data → 16 L1 misses), then all k-loop reads come from `ld.shared`. The L1 is barely used.
- **no_smem version**: 71.88% L1 hit rate. Every k-loop iteration does `ld.global.f32 %f4, [A+offset]` and `ld.global.f32 %f5, [B+offset]`. Within the 16-iteration k-loop for a given (row, col) thread, the A-element at `A[row][k]` changes each iteration (row-major, k increases → columns advance) but each cache line holds 32 floats covering multiple columns. The B-element at `B[k][col]` also changes per iteration. With 256 threads all reading the 16×16 B matrix in column-major order, the L1 captures the temporal reuse: a line loaded by one warp for iteration k is still present when other warps reach the same iteration.

## Why the smem version is faster despite 0% L1 hit rate

At first glance, 0% L1 hit rate sounds bad. But the smem version's latency breakdown is:

1. **Load phase** (once): 16 warps × 1 `ld.global` each = 16 HBM requests. These are cold misses but they complete in parallel using the 8 HBM channels.
2. **Compute phase** (16 iterations × 3 instructions): `ld.shared + ld.shared + fma.f32`. Each `ld.shared` is 1 cycle (no bank conflicts in this access pattern). The FMA follows immediately.

The no_smem version's inner loop is:
1. `ld.global.f32 %f4, [A]` — L1 hit latency 25 cycles (most of the time)
2. `ld.global.f32 %f5, [B]` — L1 hit latency 25 cycles
3. `fma.f32` — 4 cycles
4. Loop overhead (`add`, `setp`, `@p bra`) — ~4 cycles

A warp issuing those instructions takes ~25+25+4+4 = 58 cycles per k-iteration × 16 iterations = 928 cycles per warp (ignoring pipelining). The smem version's inner loop takes 1+1+4+4 = 10 cycles per k-iteration × 16 = 160 cycles.

The smem version exchanges the L1 hit latency (25 cycles per access) for the `ld.shared` latency (1 cycle per access). That is the structural advantage: **shared memory latency is ~25× lower than L1 cache latency**.

## When shared memory wins

Use shared memory when:

1. **The reuse pattern is known statically** — you can prove at compile time that a tile of data will be read multiple times by the same CTA. Matrix multiply tiles are the canonical example.
2. **The access pattern would cause L1 bank conflicts** — smem with padding or swizzled layout gives conflict-free access; L1 can only coalesce at the cache-line granularity.
3. **Latency must be predictable** — smem has no LRU variability; L1 can evict unexpectedly under pressure from other warps.

Production matmul kernels (cuBLAS, CUTLASS, FlashAttention) use shared memory exclusively for their inner loops.

## When L1 wins

Use (or rely on) L1 when:

1. **The reuse pattern is irregular** — sparse attention, graph convolution, or gather operations where you cannot predict which elements will be reused.
2. **The working set is small and fits comfortably in L1** — small matrix transposes, prefix scans.
3. **Code simplicity matters** — eliminating the tile-load + `bar.sync` infrastructure makes the kernel much simpler to write and maintain.

Some Flash-Attention variants and many inference kernels in practice rely on L1 caching of key/value tiles when the sequence length is short enough.

## 改一改 — Scale to 64×64 to break L1

Change the matrix size to 64×64. The B matrix alone is 64×64×4 = 16 384 bytes = 128 cache lines. With 256 warps accessing B in column-major order, each warp's access pattern cycles through all 128 B-matrix lines per k-step. The L1 (1024 lines total) can hold all 128 B-matrix lines simultaneously — so L1 hit rate should stay high.

Now try 256×256: B = 256 KB. That exceeds the 128 KB L1. Cache evictions begin. The no_smem variant starts losing its L1 advantage. The smem variant (with tiled loading) remains fast because each CTA only loads a 16×16 tile at a time.

## 改一改 — Shrink the L1 to 4 KB

In `default_hopper.yaml`:

```yaml
cache:
  l1_size_bytes: 4096
```

Re-run the 16×16 demo. The L1 can now hold only 32 cache lines. The B-matrix has 8 cache lines (16×16×4 / 128 = 8). With 256 threads sharing the L1, the LRU will evict B-matrix lines before the next k-iteration can reuse them. L1 hit rate drops to near 0% for the no_smem version. The smem version is unaffected — it doesn't use L1 for its inner loop at all. The cycle gap between the two variants widens dramatically.

## 真机对照 — Real-machine comparison

_No reference fixture committed (requires real-GPU run). On a real H100 with 228 KB shared memory / SM, the smem version for 16×16 matmul would use 4 KB of smem (2 × 16×16×4 B tiles) and leave 224 KB for L1. In this regime both versions would be "in cache" and the smem version would have a larger advantage due to lower smem latency. The absolute cycle counts differ (H100 runs at 1.755 GHz with more warps and pipelining), but the structural advantage of smem over L1 is the same._
