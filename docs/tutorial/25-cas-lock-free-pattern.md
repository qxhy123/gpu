# Chapter 25 — CAS & Lock-Free Pattern

## atom.global.cas: The Compare-And-Swap Primitive

`atom.global.cas` is the most expressive atomic instruction available in PTX. Where `atom.global.add` unconditionally applies an operation and returns the old value, `atom.global.cas` is conditional:

```ptx
atom.global.cas.u32 %r_ret, [%rd_addr], %r_expected, %r_desired;
```

The hardware atomically does the following:
1. Read the current value at `[%rd_addr]`.
2. If it equals `%r_expected`, write `%r_desired` to `[%rd_addr]`.
3. Return the value read in step 1 (whether or not the write happened).

The caller checks whether `%r_ret == %r_expected`. If yes, the CAS succeeded and the caller "owns" the transition from `expected` to `desired`. If no, someone else changed the memory between the read and the CAS; the caller must retry.

This single instruction is the foundation of all lock-free data structures: the **retry loop** (optimistic concurrency) replaces the acquire/release lock of a mutex.

## Why Not a Mutex Spinlock on GPU?

A traditional spinlock spins on an `ld.global` until the lock is free, then does `atom.global.cas` to acquire, runs the critical section, and writes 0 to release. This works on CPUs but **deadlocks on SIMT hardware** whenever multiple lanes of the same warp compete for the lock.

The reason: CUDA warp execution is convergent. If lane 0 holds the lock and lanes 1–31 are spinning in the acquire loop, the warp cannot make progress on the acquire path *and* the release path at the same time — it is stuck spinning. Lane 0 never reaches its release instruction because the warp is diverged: lanes 1–31 prevent convergence. The warp deadlocks.

The solution is the **CAS-retry (lock-free) pattern**: instead of a separate acquire/release protocol, each thread reads the current value, computes the desired new value, and issues a single CAS. If the CAS succeeds, the thread is done. If it fails, it re-reads and retries. In every retry round, at least one thread in the warp succeeds (the one whose read was most recent). That thread exits the retry loop. Warp convergence is restored progressively, and the warp always makes forward progress.

## 走通 atom_cas_spinlock

```bash
python examples/atom_cas_spinlock/run.py
```

The kernel uses `grid=(4,1,1)`, `block=(32,1,1)` — 128 threads total, each atomically incrementing a shared counter using the CAS-retry pattern:

```ptx
RETRY:
    ld.global.u32 %r0, [%rd0];       // read current counter
    add.u32 %r1, %r0, 1;             // desired = current + 1
    atom.global.cas.u32 %r2, [%rd0], %r0, %r1;
    setp.ne.u32 %p0, %r2, %r0;       // did CAS succeed?
    @%p0 bra RETRY;                   // no → retry
```

If the CAS returns a value different from `%r0`, another thread won the race during this round. The failing thread loops back, re-reads the updated counter, and tries again with the new expected/desired pair. Expected output:

```
atom_cas_spinlock: cycles=<N>, counter=128
```

The counter reaches exactly 128 because each of the 128 threads succeeds exactly once. The cycle count reflects the number of retry rounds needed before all threads complete.

## 看模拟器

```python
import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default

cfg = load_default()
out = np.zeros(2, dtype=np.uint32)
ptx = pathlib.Path("examples/atom_cas_spinlock/kernel.ptx").read_text()
res = gpusim.run(
    ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
    params={"OUT": out}, mode="timing", config=cfg,
)
print("cycles:", res.metrics["cycles"])
print("atomic_summary:", res.atomic_metrics.get("atomic_summary", "n/a"))
```

Open `report.html` and navigate to **§21 Atomic Events**. The CAS events will dominate the atomic event timeline. Each dot represents one CAS instruction issued; you will see many more events than there are threads because of retries. The ratio `total_cas_events / n_threads` gives you the average number of attempts per thread — a direct measure of contention on the single counter.

At high thread counts (try `grid=(8,1,1)` for 256 threads), the retry count per thread grows roughly as O(threads / hardware_parallelism), and the total cycle count grows super-linearly as more threads compete for the same address.

## 改一改

**Extend the critical section.** Replace the simple `add.u32 %r1, %r0, 1` with a longer computation — for example, an `mul` followed by an `add`. The CAS itself remains the same, but the work done between the read and the CAS attempt takes more instructions. In the simulator this does not add much time (a few extra cycles per attempt), but on real hardware it increases the window during which another thread can preempt the expected value. Compare:

- Short critical section: lower retry rate, faster completion.
- Long critical section: higher retry rate because more time passes between the read and the CAS, allowing more competing threads to succeed first.

This experiment shows why lock-free critical sections should be kept as short as possible — not to reduce the atomic latency, but to minimize the **contention window**.

## 真机对照

CUDA's standard library exposes CAS through `cuda::atomic<T>::compare_exchange_strong` and `compare_exchange_weak` from `<cuda/atomic>`. These map directly to `atom.global.cas` with the appropriate memory ordering. The `weak` variant allows spurious failures (useful in retry loops where retrying is cheap); `strong` guarantees no spurious failures.

Lock-free libraries such as **libcudacxx** implement queues, stacks, and reference-counted pointers on top of CAS. The CUDA documentation notes explicitly that mutex-style spinlocks can deadlock on SIMT hardware if multiple threads of the same warp compete — exactly the hazard described above. The CAS-retry pattern shown in `atom_cas_spinlock` is the canonical workaround and is the foundation of all production lock-free CUDA code.

For production reduction and histogram kernels, CAS-based lock-free counters are rarely the right tool — the per-address serialization overhead from many competing threads is too high. But for task queues, work-stealing schedulers, and sparse data structures where contention is low and correctness matters more than throughput, CAS-retry is indispensable.
