# compute_vs_memory_overlap

Phase 7 demo: compute-heavy kernel + memory-heavy kernel run concurrently
on two streams. Demonstrates the canonical CUDA optimization of pairing
compute and memory kernels to maximize device utilization.

## Run
```
python examples/compute_vs_memory_overlap/run.py
```

## Tutorial
docs/tutorial/28-compute-memory-overlap.md
