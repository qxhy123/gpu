# cluster_cooperative_epilogue

Phase 6 cluster TMA store cooperative epilogue demo. 4-CTA cluster: each CTA
fills its smem with rank-tagged data. CTA 0 issues 4 cluster TMA stores, each
reading from a different rank's smem (via cluster pointer encoding) and
writing to a different gmem offset.

Closes the Phase 5 cluster_matmul_dsmem deferred work: cluster TMA store
enables a single CTA to gather + store data from all cluster CTAs' smem.

Note: simplified version (no wgmma) demonstrates the cluster TMA store
mechanism; full wgmma + cooperative epilogue is the natural next step.

## Run
```
python examples/cluster_cooperative_epilogue/run.py
```

## Tutorial
docs/tutorial/24-cluster-cooperative-epilogue.md
