# Chapter 23 — smem Atomic & Bank Conflict Serialization

## Shared Memory Atomics: A Different Kind of Serialization

Chapter 22 showed how global atomics serialize through the L2 cache's atomic ALU — a unit shared across all SMs in the GPU. Shared memory atomics operate entirely within the SM, but they are not free from serialization: the bottleneck here is the **bank conflict** mechanism.

Shared memory is divided into 32 banks (on Ampere and Hopper). Any two threads accessing the same bank in the same cycle cause a bank conflict, and the hardware serializes those accesses one at a time. For regular loads and stores, bank conflicts are a recoverable latency tax. For atomics, the penalty compounds: each thread in a conflict group must wait for both the bank conflict serialization *and* the additional latency of the atomic read-modify-write operation itself (`atomic_op_extra_latency` in the simulator's model).

When all 32 threads in a warp atomically write to a single `uint32` counter at offset 0 (bank 0), every thread conflicts with every other. The hardware serializes all 32 accesses sequentially, and each one pays the atomic operation penalty on top of the bank conflict stall. The total per-warp latency grows linearly with the number of threads targeting the same bank.

This contrasts with global atomics in Chapter 22: smem atomics never leave the SM, so there is no L2 queue, no cache line locking, and no cross-SM interaction. But the per-bank serialization rule inside the SM is strictly enforced regardless.

## 走通 atom_reduction_smem

```bash
python examples/atom_reduction_smem/run.py
```

The kernel launches `grid=(1,1,1)`, `block=(128,1,1)`. Thread 0 initializes a shared memory counter at byte offset 0, all threads synchronize via `bar.sync 0`, and then every thread executes:

```ptx
atom.shared.add.u32 %r3, [%rd0], %r2;   // %rd0 = smem offset 0
```

128 threads hammering the same smem bank: 4 warps × 32 threads, each warp fully serialized. Expected output:

```
atom_reduction_smem: cycles=<N>, count=128
```

The cycle count is high relative to a simple 128-thread load, because each warp must wait for all 32 in-warp bank conflicts to drain before the next warp can proceed.

## 看模拟器

```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default

cfg = load_default()
out = np.zeros(1, dtype=np.uint32)
ptx = pathlib.Path("examples/atom_reduction_smem/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(1,1,1), block=(128,1,1),
    params={"OUT": out}, mode="timing", config=cfg,
)
print("cycles:", res.metrics["cycles"])
print("atomic_summary:", res.atomic_metrics.get("atomic_summary", "n/a"))
```

The `atomic_summary` output breaks down events by instruction type and memory space. For this kernel you will see all events tagged as `atom.shared.add` and a serialization depth of 32 (all threads in a warp hitting the same bank).

To observe the linear scaling with thread count, run the kernel with different block sizes:

| `block` | threads | expected `cycles` |
|---|---|---|
| `(32,1,1)` | 32 | baseline × 1 |
| `(64,1,1)` | 64 | ≈ 2× |
| `(128,1,1)` | 128 | ≈ 4× |

The cycle count scales proportionally with the number of warps, confirming that each warp's 32-way bank conflict is handled independently and sequentially.

## 改一改

**Spread 128 counters across 32 banks.** Replace the single counter at offset 0 with an array of 128 `uint32` counters. Assign each thread to a different counter: thread `t` writes to `smem[t]`. No two threads in a warp share a bank (since consecutive 4-byte words map to consecutive banks, and 32 consecutive words span all 32 banks exactly once).

Change the kernel's address computation to:

```ptx
mul.lo.s32 %r1, %r0, 4;    // r1 = tid * 4
cvt.u64.u32 %rd0, %r1;     // rd0 = byte offset of smem[tid]
atom.shared.add.u32 %r3, [%rd0], %r2;
```

With zero bank conflicts, each warp's 32 atomic operations proceed in parallel (one per bank). Cycle count drops to roughly 1× the single-thread latency — an ~32× improvement over the single-counter case. The final result must then be reduced across all 128 partial counters to get the total.

This is the core idea behind **tiled reduction**: use private per-thread or per-warp accumulators in smem, then merge them in a tree reduction. The `reduction_smem` example (Chapter 4) covers that final merge step.

## 真机对照

On real hardware (Ampere / Hopper), `atom.shared` instructions are handled entirely within the SM's shared memory hardware, not by the L2. The SM contains dedicated atomic processing logic per bank: when a conflict occurs, the hardware queues the accesses and processes them one per clock per bank. There is no way to issue the same smem atomic operation to two threads sharing a bank "in parallel" — the hardware guarantee of atomicity requires strict serialization.

Nsight Compute reports this under **Shared Memory Bank Conflicts** → look for "bank conflict stall cycles" in the warp stall reasons panel. A high `LG Throttle` or `MIO Throttle` stall percentage alongside a high bank conflict count is the signature of the single-counter pattern.

The fix — privatization followed by tree reduction — is universal. CUDA's `cooperative_groups::reduce` and CUB's `BlockReduce` both implement it automatically. Understanding the bank conflict serialization model in the simulator gives you an intuitive explanation for *why* privatization is so effective: it turns O(threads) serial atomics into O(1) parallel stores followed by O(log threads) conflict-free tree steps.
