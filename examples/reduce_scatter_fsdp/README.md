# reduce_scatter_fsdp

Phase 12 demo: 4-GPU FSDP-style reduce_scatter on 256-byte grads.
Each rank gets 1/4 of reduced result. Demonstrates N-1 = 3 NVLink transfers per rank.

## Run
```
python examples/reduce_scatter_fsdp/run.py
```

## Tutorial
docs/tutorial/48-reduce-scatter-fsdp.md
