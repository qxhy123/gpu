# tree_allreduce

Phase 10 demo: 4-GPU tree allreduce on a 64-byte buffer (small message →
tree path auto-picked). 2*log2(N) = 4 transfer steps for N=4.

## Run
```
python examples/tree_allreduce/run.py
```

## Tutorial
docs/tutorial/42-tree-allreduce-latency-optimal.md
