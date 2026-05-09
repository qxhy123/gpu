# Chapter 22 — gmem Atomic & L2 ALU Serialization

## Global Memory Atomics Route Through the L2 Atomic ALU

When a thread issues `atom.global.add` (or any other global atomic), the request does not touch the L1 cache at all. It travels straight to the L2 cache, where a dedicated **atomic ALU** performs the read-modify-write in place. The L2 atomic ALU operates on cache-line granularity: it locks the line, reads the current value, applies the operation, writes the new value back, and unlocks — all within L2 without evicting data to HBM.

The critical implication: **two atomics targeting the same L2 cache line cannot proceed simultaneously**. The second request must wait in a per-line FIFO queue until the first finishes. The simulator models this with an `L2AtomicQueue` attached to each cache line, serializing arrivals and recording the queue depth at each cycle.

With 132 SMs on an H100, all issuing atomic instructions in the same cycle, a single "hot" address can see a queue depth of 132 in the worst case. The serialization latency multiplies with each additional waiter: if one atomic takes 20 ns, a queue of 132 means the last thread waits over 2.6 µs — for a single instruction.

## 走通 atom_histogram

```bash
python examples/atom_histogram/run.py
```

The kernel launches `grid=(8,1,1)`, `block=(32,1,1)` — 256 threads total. Each thread selects a bin by masking the low 4 bits of its thread ID (`and.b32 %r2, %r1, 15`), then issues:

```ptx
atom.global.add.u32 %r5, [%rd2], %r4;
```

With 16 bins and 256 threads, each bin receives on average 16 hits. Every hit from a different SM must serialize through the L2 atomic ALU for that bin's cache line. Expected output:

```
atom_histogram: cycles=<N>
  bins = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16]
```

The cycle count reflects the serialization overhead from 16-way contention per bin. Compare this baseline to what happens at higher and lower collision rates (see 改一改 below).

## 看模拟器

Run in timing mode with the Python API and inspect the atomic metrics:

```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default

cfg = load_default()
out = np.zeros(16, dtype=np.uint32)
ptx = pathlib.Path("examples/atom_histogram/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(8,1,1), block=(32,1,1),
    params={"OUT": out, "N_BINS": 16}, mode="timing", config=cfg,
)
print("cycles:", res.metrics["cycles"])
print("serialization_overhead:", res.atomic_metrics["serialization_overhead"])
print("max_queue_depth:", res.atomic_metrics["max_queue_depth"])
```

Open `report.html` and navigate to **§21 Atomic Events**:

- The **hot line** panel shows which L2 cache lines accumulated the most atomic requests. With 16 bins packed into a small array, only 1–2 cache lines are hot.
- The **queue depth timeline** shows each line's FIFO depth over simulation time. A depth of 16 means 15 threads are waiting while 1 is being served.

The `serialization_overhead` metric reports total thread-cycles spent waiting in the L2 atomic queue. Higher values here directly explain cycle count inflation compared to a no-contention baseline.

## 改一改

**Fewer bins → severe contention.** Change the kernel's bin selection to `and.b32 %r2, %r1, 3` (4 bins instead of 16). Now 64 threads share each bin, and queue depth jumps to 64 per bin's L2 line. Cycle count roughly quadruples compared to the 16-bin baseline.

**More bins → near-zero serialization.** Change to `and.b32 %r2, %r1, 127` (128 bins). With only 2 threads per bin on average, the L2 atomic queue rarely has more than 2 waiters. Cycle count drops close to a single non-contended atomic latency.

This experiment cleanly isolates the serialization cost from everything else: the number of memory instructions is identical in all three configurations; only the collision rate changes.

## 真机对照

On an H100, the L2 cache includes a limited number of atomic ALU units — each capable of processing one atomic per cycle per bank. When thousands of threads from 132 SMs converge on a small number of global addresses (a common pattern in reduction kernels, lock-free queues, or hash tables with many collisions), the L2 atomic ALU becomes the bottleneck. NVIDIA profiles this as "L2 atomic serialization" in Nsight Compute under the **Memory Workload Analysis** section.

Production kernels that accumulate into a histogram-like structure solve this by:

1. **Partial histograms in smem** — each CTA keeps its own counters in shared memory (no L2 contention), then merges into global memory once per CTA (low contention).
2. **Privatization** — allocating a separate output array per SM or per warp and doing a final sweep.

The `atom_histogram` example is intentionally the worst case. Once you understand what the L2 queue depth means in the simulator, you can reason about the same hot-key pathology in production workloads like graph traversal, sparse attention, or any algorithm with skewed access distributions.
