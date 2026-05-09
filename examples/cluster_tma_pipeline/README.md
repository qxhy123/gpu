# cluster_tma_pipeline

Phase 5 cluster TMA + dsmem demo. 4-CTA cluster: CTA 0 loads full 256-elem
buffer via TMA into its smem_T; cluster barrier signals completion; each CTA
reads its 64-elem slice via `mapa.shared::cluster` + `ld.shared::cluster`;
writes to OUT.

Each of the 32 threads per CTA handles 2 elements (elem at rank*64+tid and
rank*64+tid+32), covering all 64 elements in the CTA's slice.

Demonstrates: TMA in cluster context + cluster barrier + dsmem ld.

## Run
```
python examples/cluster_tma_pipeline/run.py
```

## Tutorial
docs/tutorial/21-cluster-tma-pipeline.md
