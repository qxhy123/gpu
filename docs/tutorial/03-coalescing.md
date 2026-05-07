# Chapter 03 — Global Memory Coalescing

## What "coalesced" really means

NVIDIA GPUs load global memory in units called **sectors** (128 bytes). A sector is roughly a cache-line boundary. When 32 threads in a warp issue a `ld.global`, the hardware groups the 32 addresses and counts how many distinct sectors they touch.

- **Best case (fully coalesced):** all 32 threads access addresses within the same 128-byte sector. One transaction fetches the data for all 32 lanes.
- **Worst case (fully strided):** each thread accesses a different sector. 32 transactions are needed.

The ratio `useful_bytes / fetched_bytes` is the **coalescing efficiency**. For float32 loads: best case = 32 × 4 = 128 bytes used / 128 bytes fetched = 1.0. Worst case = 128 bytes used / (32 × 128) = 1/32 ≈ 0.031.

**Why it matters:** global memory bandwidth is the limiting resource for most GPU workloads. A kernel that reads 4× more data than necessary is 4× slower, with no instruction-count benefit.

## Walking `coalescing_demo` for different strides

The `examples/coalescing_demo/kernel.ptx` kernel reads `A[tid * STRIDE]` and writes `OUT[tid]`. With `STRIDE=1`, consecutive threads access consecutive elements — perfect coalescing. With `STRIDE=2`, each thread accesses every other element — 50% efficiency. And so on.

Run the demo:

```bash
python examples/coalescing_demo/run.py
```

Output:

```
stride=1: cycles=422
stride=2: cycles=423
stride=4: cycles=425
stride=8: cycles=429
```

The cycle count **does increase with stride** in the simulator's Phase 1 model. This is because the simulator scales LSU issue occupancy by `n_transactions`: a stride-1 load requires 1 transaction and occupies the LSU for 1 cycle, while a stride-8 load requires 8 transactions and occupies the LSU for 8 cycles. In addition, each extra transaction adds one cycle of result latency (to model burst completion — the last transaction completes one cycle after the one before it). For this kernel, which has a single global load per warp, the deltas are modest (422 → 423 → 425 → 429), but in a realistic kernel with many global loads the effect compounds significantly.

What *also* changes is the coalescing analysis metric in the HTML report — and that is the primary teaching signal.

Open `report.html` after running `STRIDE=1` vs `STRIDE=4`: the **coalescing report** section shows the number of 128-byte transactions for the `ld.global` instruction. For `STRIDE=1` you get 1 transaction for 32 threads; for `STRIDE=4` you get 4 transactions; for `STRIDE=32` (accessing every 32nd element) you get 32 transactions.

To verify programmatically:

```python
import numpy as np, gpusim, pathlib

ptx = pathlib.Path("examples/coalescing_demo/kernel.ptx").read_text()
a = np.arange(1024, dtype=np.uint32)

for stride in [1, 2, 4, 8, 32]:
    out = np.zeros(32, dtype=np.uint32)
    res = gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                     params={"A": a, "OUT": out, "STRIDE": stride},
                     mode="functional")
    # OUT[i] = A[i * stride]
    expected = a[:32*stride:stride]
    assert np.array_equal(out, expected), f"stride={stride} failed"
```

The assertion passes for all strides — the *correct* values are always produced, just with different efficiency.

## The `n_transactions` column

In Phase 1, coalescing info is collected during timing mode and reported per `ld.global` instruction. The key metric is `n_transactions`: how many 128-byte sector fetches were needed to satisfy the 32-lane load.

For a `ld.global.u32` (4-byte load) with 32 consecutive addresses starting at a 128-byte-aligned base:
- All 32 addresses fit in one 128-byte sector → `n_transactions = 1`.

For stride 4 (addresses are 16 bytes apart, i.e., 4 × 4 bytes):
- Thread 0: byte offset 0, Thread 7: byte offset 112 — these 8 fit in the first sector (0–127).
- Thread 8: byte offset 128, … — next sector.
- Total 4 sectors → `n_transactions = 4`.

The HTML report's coalescing section shows a histogram of `n_transactions` across all global memory accesses. A well-coalesced kernel shows a spike at 1; a poorly-coalesced one shows a spread toward 32.

## 改一改 — Change dtype from `u32` to `u64`

In `kernel.ptx`, change `ld.global.u32` to `ld.global.u64` and adjust the shift from `shl.b32 %r4, %r3, 2` (×4) to `shl.b32 %r4, %r3, 3` (×8). Now each element is 8 bytes.

With `STRIDE=1`, 32 threads × 8 bytes = 256 bytes = 2 sectors → `n_transactions = 2`. With `STRIDE=2`, 32 threads × 16-byte gaps = 4 sectors (threads 0–7 in sector 0, 8–15 in sector 1, 16–23 in sector 2, 24–31 in sector 3, but wait — with 8-byte elements and stride 2, the stride in bytes is 16 bytes, so 32 threads span `31 * 16 + 8 = 504 bytes` → 4 sectors). The efficiency halves again compared to the `u32` case.

The general rule: **the effective stride in bytes** determines the sector span. Larger dtypes make the same logical stride worse in byte terms.

## Phase 1 limitations

Phase 1 models `n_transactions` as a direct LSU occupancy and latency multiplier — a simplified approximation that captures the qualitative cost of poor coalescing. Real GPUs use more complex memory pipelines: sector merging, write combining, and hardware coalescing at the L1/L2 boundary can change the effective transaction count at runtime. The Phase 1 model errs on the side of counting every transaction as a serialized LSU cycle, which overestimates the cost slightly but correctly orders the strides by cycle count.

Phase 1 does not model:
- **Bandwidth contention** — multiple CTAs competing for the same DRAM channels.
- **L1/L2 cache** — in reality, coalesced accesses often hit the L1 cache after the first pass, dramatically reducing effective transactions.
- **Hardware prefetching** — real GPUs prefetch ahead of demand loads.

The `n_transactions` LSU occupancy scaling is documented in Spec §11/§12 as the Phase 1 trade-off. Phase 2 will refine this with a full cache hierarchy and bandwidth queue, at which point cycle counts will more accurately reflect memory system behavior.
