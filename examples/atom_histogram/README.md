# atom_histogram

Phase 6 gmem atomic demo. 8 CTAs × 32 threads each atomic.add to a bin
determined by `tid & 15`. With n_bins=16 and 32 threads/CTA, each bin sees
2 atomic per CTA × 8 CTAs = 16 atomic / bin.

L2 atomic ALU serializes atomic on same line; high contention → high latency.

## Run
```
python examples/atom_histogram/run.py
```

## Tutorial
docs/tutorial/22-gmem-atomic-l2-alu.md
