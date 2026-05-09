# atom_reduction_smem

Phase 6 smem atomic demo. 128 threads each atomic.add 1 to a single smem counter.
All 128 atomic ops serialize through one bank → high latency.

## Run

```
python examples/atom_reduction_smem/run.py
```

## Tutorial

docs/tutorial/23-smem-atomic-bank-conflict.md
