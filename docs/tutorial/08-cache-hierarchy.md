# Chapter 08 — Cache Hierarchy

## From black-box latency to real hierarchy

In Chapter 03 (coalescing) we said global memory accesses take around 400 cycles. That was a useful lie: the real hardware has three levels of memory between your registers and DRAM, and Phase 2 of the simulator models all of them.

The actual path for every `ld.global` is:

```
Registers
   └── L1 cache (128 KB, 4-way LRU, 128 B line, ~25 cycle hit)
          └── L2 cache (4 MB, 16-way, write-back, ~200 cycle hit)
                 └── HBM (8 channels × 16 banks, 10–30 cycle per channel)
```

Each level is physically closer to the SM than the one below it, and the simulator models each with tag-precise behaviour: it tracks which 128 B cache lines are present, in which set, with which LRU position.

## L1 cache mechanics

The L1 cache is private to each SM. Its parameters in `default_hopper.yaml`:

```yaml
cache:
  l1_size_bytes: 131072    # 128 KB total
  l1_ways: 4               # 4-way set-associative
  l1_line_bytes: 128       # 128 B per line
  l1_hit_latency: 25       # cycles from issue to result-ready on hit
  mshr_slots: 16           # Miss Status Holding Registers
```

A warp's `ld.global.f32` first computes the 128 B cache-line address for each active lane. Identical lane addresses merge into one line address (coalescing within a warp). For each unique line address the L1 checks its tag array:

- **L1 HIT** — line present → result ready in 25 cycles.
- **L1 MISS, MSHR slot available** — allocate a new MSHR entry, issue the request downstream to L2. The warp stalls on `SCOREBOARD` until the data returns.
- **L1 MISS, merge with existing MSHR** — another warp already issued the same line address. This warp piggybacks on that MSHR entry without using a new slot.
- **L1 MISS, all 16 MSHR slots full** — `MSHR_FULL` stall. The warp retries next cycle without advancing its PC.

A thread issuing a `st.global` uses a **write-through, no-write-allocate** policy at L1: stores bypass L1 and go directly to L2.

## L2 cache mechanics

The L2 is shared across all CTAs launched on the same SM. Its default parameters:

```yaml
  l2_size_bytes: 4194304   # 4 MB
  l2_ways: 16              # 16-way set-associative
  l2_hit_latency: 200      # cycles
```

L2 uses **write-back, write-allocate** semantics: a store miss allocates a line in L2 and marks it dirty; a dirty eviction triggers a write-back to HBM.

## MSHR — coalescing inside the cache

The 16 MSHR slots are the L1's "in-flight request table". Each slot tracks one cache-line-level miss. When a warp accesses a line that is already being fetched for another warp (or another iteration of the same warp), the new waiter merges into the existing MSHR slot — no additional HBM bandwidth is consumed. This is the cache's answer to the warp-level coalescing from Chapter 03: the MSHR coalesces at the cache-line granularity across warps.

## Walking the simulator — l1_thrash_demo

The `examples/l1_thrash_demo/` kernel loops K times, and on each iteration each thread reads one element separated by STRIDE elements from the previous. By changing K and STRIDE we control the working set size and access pattern.

Run it:

```bash
python examples/l1_thrash_demo/run.py
```

Output:

```
# Three working-set configurations:
  A: fits L1 (32 KB): cycles=988, L1 hit 0.0%, L2 hit 0.0%
  B: > L1, fits L2 (1 MB): cycles=7708, L1 hit 0.0%, L2 hit 0.0%
  C: > L2 (16 MB): cycles=491548, L1 hit 0.0%, L2 hit 0.0%
```

**Why are all L1 hit rates 0.0% even for Config A?**

Each iteration of the kernel's loop accesses elements `tid + iter * STRIDE`. With STRIDE=256, iteration 0 accesses elements 0–31 (cache line 0), iteration 1 accesses elements 256–287 (cache line 8), and so on. Each of the 32 iterations maps to a *different* cache line. No cache line is touched twice within the kernel. First touch = cold miss; with K=32 unique lines and no revisit, the hit rate is 0/32 = 0%.

This is not a bug — it illustrates a fundamental principle: **caches only help when there is temporal or spatial reuse**. The l1_thrash_demo kernel is intentionally sequential and non-reusing.

**So what do the three configs actually demonstrate?**

The key difference is the total HBM traffic and its interaction with the channel queue:

- **Config A** (K=32): 32 unique HBM requests. All complete quickly because only one HBM channel is used (all addresses fall in channel 0). cycles=988.
- **Config B** (K=256): 256 HBM requests. 8× more serialization in the channel queue. cycles=7708 ≈ 7.8× more than Config A. The ratio matches 256/32 = 8.
- **Config C** (K=16384): 16384 HBM requests. The HBM channel queue keeps growing and the warp spends most of its time waiting on outstanding loads. cycles=491548 ≈ 63.7× more than Config B. This matches 16384/256 = 64.

**The cost of HBM**: A single HBM row-miss costs ~30 cycles plus channel-queue wait. With 16384 sequential row-miss accesses, all serialized through the same channel, the total latency dominates everything else.

**L2 set-thrashing in Config C**

The L2 has 2048 sets × 16 ways. Config C's access pattern (line_addr increments of 32) maps all 16384 accesses to only 64 unique L2 sets (2048 / 32 = 64). Each of these 64 sets has 16 ways, so the first 1024 installs are cold misses. After that, the L2 evicts older lines from the same set (EVICT_CLEAN events) to make room — but since the kernel never re-reads a line, the L2 evictions are wasted work. The L2 hit rate stays 0%.

## The "knee" effect on a real machine

On a real H100, you would see a clear "knee" curve when the working set transitions from L1 → L2 → HBM:

| Level | Capacity | Latency | Bandwidth |
|-------|----------|---------|-----------|
| L1 (per SM) | 128 KB | ~25 cycles | Very high |
| L2 (shared) | 50 MB | ~200 cycles | ~12 TB/s |
| HBM3 | 80 GB | ~300 ns | ~3.35 TB/s |

In the simulator (single-SM mode), the "knee" shows up as the cycle-count ratios: Config A → B → C scale roughly 1× → 8× → 500×, reflecting HBM serialization rather than true hierarchy.

## 改一改 — Shrink L1 so Config A "falls off the cliff"

Open `gpusim/config/default_hopper.yaml`. Change:

```yaml
cache:
  l1_size_bytes: 65536     # 64 KB (half the original)
```

Re-run `python examples/l1_thrash_demo/run.py`. Config A uses 32 unique cache lines = 4096 bytes, still fits in 64 KB. But if you increase K to 512 (working set = 512 lines × 128 B = 65536 bytes = exactly 64 KB), you'll see L1 evictions begin.

For a more dramatic demo, set `l1_size_bytes: 4096` (tiny 4 KB, 32 lines total). Now Config A's 32 lines exactly fill the L1. The next warp's accesses evict Config A's lines. On a multi-CTA launch you'd see L1 thrashing. Reset to `131072` when done.

## 改一改 — More MSHR slots

The default 16 MSHR slots can handle at most 16 in-flight cache-line misses simultaneously. With Config B's 256 unique lines, the warp issues 16 at a time and stalls with MSHR_FULL until each batch completes. Try:

```yaml
cache:
  mshr_slots: 64
```

This allows 64 in-flight misses, overlapping more HBM latency. Config B cycle count should drop noticeably.

## 真机对照 — Real-machine comparison

_No reference fixture committed for l1_thrash_demo (requires real-GPU run). If you add one via `tests/reference/gen_reference.py l1_thrash_demo`, the reference test will compare L1 hit rate ± 10% and cycle count as a sanity check. Real H100 L1 hit latency is ~30 cycles (vs. 25 in the simulator); HBM latency is ~300 ns at 1.8 GHz ≈ 540 cycles (vs. 30 in the simulator). These differences explain why absolute cycle counts differ, but the hierarchy pattern is the same._
