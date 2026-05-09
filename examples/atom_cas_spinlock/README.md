# atom_cas_spinlock

Phase 6 CAS lock-free pattern demo. N threads increment a shared counter via
`atom.global.cas.u32` retry loop (compare-and-swap based lock-free increment).

Demonstrates: CAS retry loop pattern; SIMT-safe lock-free critical section.

## Design note: CAS spinlock vs CAS retry

A traditional mutex-style spinlock (`atom.cas` to acquire → load/add/store → release)
causes **SIMT deadlock** in this simulator when multiple lanes of the same warp
compete for the lock.  When lane 0 acquires the lock, lanes 1–31 spin at the
top of the SIMT stack — and can never make progress because lane 0 (which would
release the lock) is sitting below them in the stack, waiting to be scheduled.

The kernel here avoids this by using a **lock-free CAS retry loop** instead:

```
RETRY:
    ld counter → old
    new = old + 1
    atom.cas(counter, old, new) → got
    if got != old: goto RETRY
```

This pattern is safe under SIMT divergence: in each retry round, at least one
lane succeeds and exits the loop, so the warp always converges.

## Run

```
python examples/atom_cas_spinlock/run.py
```

## Tutorial

docs/tutorial/25-cas-lock-free-pattern.md
