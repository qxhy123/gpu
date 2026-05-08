# Chapter 11 — DRAM Row Buffer Locality

## DRAM is electrically different from SRAM

SRAM (used for L1, L2, shared memory, and register file) stores each bit in 6 transistors. Each read is non-destructive and takes a few cycles. DRAM stores each bit in 1 capacitor. Reading it destroys the value — the row must be amplified first (ACTIVATE), read (READ), then restored (PRECHARGE). This three-step "row cycle" takes tens of nanoseconds.

The saving grace: DRAM banks have a **row buffer**. When a row is activated, the entire 4 KB row is latched into the row buffer. Subsequent accesses to the same open row cost only the READ step, not the ACTIVATE+READ+PRECHARGE cycle. This is a row buffer **hit** — roughly 3× faster than a row buffer miss in real hardware.

The simulator captures this with two parameters:

```yaml
hbm:
  row_size_bytes: 4096    # 4 KB row
  row_hit_latency: 10     # cycles (same open row)
  row_miss_latency: 30    # cycles (must open new row)
```

## Address layout caveat — why stride = row_size is not enough

Intuition says: "each 4 KB row holds 1024 float32 elements, so a stride of 1024 elements jumps to the next row and causes row misses." This intuition is wrong for the simulator's address layout.

The bit layout in the Phase 2 simulator is:

```
Byte address bits:
  [6:0]  = offset within cache line (128 B)
  [9:7]  = channel index (3 bits, 8 channels)
  [14:10] = column within DRAM row (5 bits, 32 columns × 128 B = 4 KB row)
  [18:15] = bank index (4 bits, 16 banks)
  [30:19] = row index (12 bits)
```

To jump from one row to the next row **in the same bank** you need to increment bits [30:19] while holding bits [18:0] constant. That means advancing the byte address by `2^19 = 524288 bytes = 131072 float32 elements`. A stride of 1024 elements only advances bits [14:10] (the column within the row) — it stays in the same row.

The actual stride needed to guarantee row misses is `131072 floats` (for all threads to access different rows of the same bank). However, this stride causes all 32 threads to access the same L1 cache set (because `131072 × 4 / 128 = 4096`, a multiple of L1's 256 sets), causing L1 cache-set thrashing that makes the simulation extremely slow.

## The practical demonstration

`examples/row_buffer_demo/` uses a kernel that reads one element per thread with a configurable `STRIDE`. Two strides show contrasting row-buffer behavior without causing cache-set thrashing:

```bash
python examples/row_buffer_demo/run.py
```

Output:

```
# stride=32 (within-row, row hits dominate):
  cycles=121, row_buffer_hit_rate=72.73%
# stride=65568 (cross-row stride, row misses dominate):
  cycles=163, row_buffer_hit_rate=0.00%
```

**Stride=32 (within-row access):**
Thread tid accesses element `tid × 32`. These are 32 different 128 B cache lines (each thread in its own line), but all in row 0 of their respective channels. The first access to each channel opens row 0 (ROW_MISS). The second access to the same channel's open row 0 is ROW_HIT. With 32 threads cycling through 8 channels 4× each, roughly 3 of 4 accesses hit the open row → 72.73% row buffer hit rate.

**Stride=65568 (cross-row access):**
Stride 65568 floats = 262272 bytes = line_addr increment of 2049. The HBM row field increments by `2049 / 512 ≈ 4` rows per thread (different rows and different L1 sets). Each thread accesses a unique row. No row is opened more than once → 0% row buffer hit rate.

The cycle difference: 121 vs. 163. Row hits (10 cycles each) vs. row misses (30 cycles each) account for ~35% more time for the all-miss case.

## Reading the row buffer hit rate

```python
import numpy as np, pathlib, gpusim

ptx = pathlib.Path("examples/row_buffer_demo/kernel.ptx").read_text()
n = 16 << 20
a = np.arange(n, dtype=np.float32)

out = np.zeros(32, dtype=np.float32)
res = gpusim.run(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                 params={"A": a, "OUT": out, "STRIDE": 32}, mode="timing")

print("row_buffer_hit_rate:", res.cache_metrics["row_buffer_hit_rate"])
print("HBM events breakdown:")
print(res.hbm_events_df[["channel", "bank", "row", "row_kind"]].head(12).to_string())
```

Output:

```
row_buffer_hit_rate: 0.7272727272727273
HBM events breakdown:
   channel  bank  row  row_kind
0        0     0    0  ROW_MISS
1        1     0    0  ROW_MISS
2        2     0    0  ROW_MISS
3        3     0    0  ROW_MISS
4        4     0    0  ROW_MISS
5        5     0    0  ROW_MISS
6        6     0    0  ROW_MISS
7        7     0    0  ROW_MISS
8        0     0    0   ROW_HIT
9        1     0    0   ROW_HIT
10       2     0    0   ROW_HIT
11       3     0    0   ROW_HIT
```

The first 8 events open row 0 in channels 0–7 (ROW_MISS each). The next 24 events are ROW_HIT — the same row 0 is still open in each channel. The stores to `OUT` follow the same channel cycling pattern, adding more hits.

## Why AI workloads care about row buffer locality

Matrix operations in transformer models naturally exhibit row-major sequential access. When a kernel iterates over a row of a large matrix (e.g., `A[row, :]` for `row = 0, 1, 2, ...`), successive column accesses stay within the same DRAM row. Each row is opened once and then hit many times.

Contrast with **column-major access** in a non-transposed matmul: accessing `A[:, col]` jumps by `(number of columns × 4 bytes)` between successive elements. If columns × 4 bytes > 4096 (one DRAM row), every access misses the row buffer.

This is one reason cuBLAS matrices are stored in Fortran (column-major) order: it ensures the inner loop of the fastest-varying index stays within the same DRAM row.

## 改一改 — Stride within row vs. stride across rows

Try additional stride values to build intuition:

**stride=1** (fully coalesced): All 32 threads access elements 0–31 — a single 128 B cache line. One HBM request is made. That's 1 ROW_MISS followed by 0 more accesses to the same bank. row_buffer_hit_rate = 0% (only 1 event, a miss). But this is the most efficient access pattern — minimum HBM traffic.

**stride=4** (quarter-line stride): Elements 0, 4, 8, ..., 124. Threads 0–31 cover bytes 0–124 — still within 1 cache line. Same as stride=1 from the L1/HBM perspective.

**stride=32** (line stride): Each thread in a different 128 B line, all in row 0 of their channel. Multiple lines from the same row → ROW_HIT after the first. This is what the demo uses.

**stride=1024** (row stride in terms of columns): Elements 0, 1024, 2048, .... The column field increments by `1024 × 4 / 128 = 32` per thread. With 32 threads, you wrap around col 0–31 multiple times. All still in row 0! row_buffer_hit_rate stays high.

The jump to row misses requires striding across rows (bits [30:19]), which needs a minimum stride of 131072 floats given the bit layout.

## 改一改 — Change row_hit_latency and row_miss_latency

In `default_hopper.yaml`:

```yaml
hbm:
  row_hit_latency: 5    # half the default
  row_miss_latency: 100 # much longer miss
```

Re-run the demo. The cycle ratio between stride=65568 (all misses) and stride=32 (mostly hits) should increase dramatically: stride=65568 pays 100 cycles per miss, stride=32 pays 5 cycles per hit. Row buffer locality becomes 10× more important than at the defaults.

Real HBM3 has typical row cycle times of:
- tRCD (ACTIVATE to READ): ~14 ns
- tCL (READ latency): ~14 ns  
- tRP (PRECHARGE): ~14 ns

At 1.8 GHz = 0.56 ns/cycle, these translate to ~25 cycles each. Row hit (just READ) ≈ 14 ns ≈ 25 cycles. Row miss (ACTIVATE + READ + PRECHARGE) ≈ 42 ns ≈ 75 cycles. The simulator's values (10 and 30) are compressed but preserve the 3× ratio.

## 真机对照 — Real-machine comparison

_No reference fixture committed (requires real-GPU run). Real H100 HBM3 uses **XOR address hashing** to interleave rows across banks and channels in a way that avoids power-of-2 aliasing. This makes the mapping from virtual address to DRAM row non-trivial, and stride=131072 floats does NOT guarantee row misses on the real machine. The simulator's simpler linear address layout makes the row-miss stride predictable at the cost of accuracy for complex access patterns. The qualitative lesson — sequential same-row access = fast, row-crossing access = slow — applies to both._
