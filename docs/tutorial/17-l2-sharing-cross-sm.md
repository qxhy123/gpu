# Chapter 17 — L2 共享与 Cross-SM Coalescing

## 共享 L2 的设计动机

When a single SM misses in its L1 cache, it sends a request to L2. With a multi-SM topology, multiple SMs can independently miss on the *same* cache line at the *same* time. Naive behavior would send two separate HBM fetch requests for the same address — doubling the memory bandwidth consumed and the latency paid.

The L2's MSHR (Miss Status Holding Register) prevents this. An MSHR is a buffer slot that tracks an in-flight L2 miss: when the first request for a cache line arrives, an MSHR slot is allocated and the HBM fetch is issued. If a second SM misses on the same line before the fetch completes, its request is *merged* into the existing MSHR entry — it waits for the same fetch to return. When the data arrives from HBM, both SMs are satisfied with a single memory transaction.

This is **cross-SM coalescing at the L2 level**. It is distinct from warp-level coalescing (Chapter 3, in-warp address merging before hitting L1) and from L1 hit sharing (two warps on the *same* SM can share an L1 line). Cross-SM coalescing only matters once you have multiple SMs targeting the same memory region — which happens whenever CTAs share read-only input data, weight tensors, or lookup tables.

## L2 MSHR 与 Cross-SM Hit 指标

The simulator tracks each L2 cache line's **origin SM** — the SM whose request first triggered the HBM fetch. When a second SM hits on that line while it is still in L2 (or being fetched), the simulator records a **cross-SM hit**: the accessing SM is different from the origin SM.

Two key metrics exposed in `res.cache_summary()`:

- `l2_cross_sm_hit_rate`: fraction of L2 accesses satisfied by a line whose `origin_sm` differs from the requesting SM.
- `l2_mshr_conflicts`: count of requests that had to wait because no MSHR slot was free (all 32 default slots occupied).

The demo kernel in `examples/l2_sharing_demo/kernel.ptx` constructs a deliberate sharing pattern. With 8 CTAs (one per SM), each CTA reads from a small window of `RO_IN` that overlaps significantly with its neighbors:

```ptx
// read index = ctaid*8 + tid  (window of 32 elements, stride 8)
shl.b32 %r2, %r0, 3;       // ctaid * 8
add.s32 %r2, %r2, %r1;     // + tid
```

CTA 0 reads indices 0–31; CTA 1 reads indices 8–39; CTA 2 reads indices 16–47. The windows overlap by 24 elements (75%) with each neighbor. L2 cache lines covering indices 8–31 are needed by multiple SMs, creating cross-SM sharing opportunities.

The write path uses non-overlapping windows (`ctaid*32 + tid`), so writes have no sharing — this isolates the cross-SM effect to the read side.

## 走通 l2_sharing_demo

```bash
python examples/l2_sharing_demo/run.py
```

Expected output:

```
l2_sharing_demo: cycles=820
  cache_summary: {'l1_hit': 0, 'l1_miss': 256, 'l2_hit': 192, 'l2_miss': 64,
                  'l2_cross_sm_hit': 168, 'l2_mshr_conflicts': 0}
```

Reading the numbers:
- 256 total L1 misses (8 SMs × 32 threads × 1 access each).
- 192 L2 hits — most of the L1 misses find the data already in L2 because a neighboring SM already fetched it.
- 64 L2 misses go to HBM — these are the "first touch" requests that each miss before any other SM has loaded that line.
- 168 of the 192 L2 hits are **cross-SM hits**: the requesting SM is different from the SM that originally fetched the line.
- 0 MSHR conflicts: 32 MSHR slots is more than enough for 8 SMs with this access pattern.

The `l2_cross_sm_hit_rate` is 168 / 256 = 65.6%. Nearly two-thirds of all memory accesses are satisfied by cross-SM reuse — no redundant HBM traffic.

Open `report.html` and navigate to **§17 L2 MSHR events**. Each row in the MSHR table shows:
- The physical address of the cache line being fetched.
- The origin SM (first to request it).
- The cycle range from HBM fetch start to data return.
- Any additional SMs that merged into the same MSHR slot (cross-SM waiters).

## 改一改

**Saturate the MSHR:**

```python
import gpusim
from gpusim.config.loader import load_default

cfg = load_default()
cfg.cache.l2_mshr_slots = 8   # reduce from 32 to 8
```

With only 8 MSHR slots and 8 SMs each issuing concurrent misses, the MSHR fills immediately. The `l2_mshr_conflicts` count jumps sharply; some SM requests must stall waiting for a slot to free up before they can even register their miss. Total cycle count increases as SM warps are stalled by MSHR backpressure rather than just HBM latency.

**Remove overlap:** Change the read index formula from `ctaid*8 + tid` to `ctaid*32 + tid` (non-overlapping windows, same as the write side). Now each SM reads a completely disjoint region. Cross-SM hits drop to zero; all L2 misses must be satisfied from HBM. Total cycles increase because the full HBM latency is paid for every access.

**Increase SM count:**

```python
cfg.n_sm = 16
```

With 16 SMs and the same overlapping pattern, the middle SMs (those with many neighbors on both sides) see even higher cross-SM hit rates. SMs at the edges of the index range have fewer neighbors and lower hit rates. Observe how the per-SM breakdown in the HTML report shows this gradient.

## 真机対照

The H100 SXM5 has **60 MB of L2 cache** organized into **12 slices**, each with its own MSHR structure. Key facts:

- Each L2 slice services requests from all 132 SMs, with cross-SM coalescing at the slice level.
- The per-slice MSHR count is not publicly documented but is estimated at 64–128 slots based on microbenchmarks.
- The H100's L2 supports **shared atomics across SMs** (`atom.global.sys`) which depend on the same L2 MSHR infrastructure — concurrent atomics to the same address are serialized inside the MSHR.
- Cross-SM L2 sharing is critical for multi-CTA reductions, broadcast operations (e.g., loading the same bias tensor from multiple CTAs), and any kernel where multiple CTAs read overlapping input tiles (common in batched GeMMs with shared weight matrices).

The simulator's single-L2 model captures the MSHR coalescing effect accurately. The main simplification is the absence of L2 slice-level routing: in hardware, an address maps to a specific slice based on its page coloring, and cross-SM conflicts only arise within the same slice. In the simulator, all SMs compete in a single MSHR pool.
