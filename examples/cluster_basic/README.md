# cluster_basic

Phase 5 minimal Hopper cluster: 2 CTAs in a cluster. Each CTA writes its
ctaid.x to OUT, then synchronizes via `barrier.cluster.{arrive,wait}`.

Smallest example demonstrating cluster co-residency + barrier semantics.
T12-T13 will extend with `mapa.shared::cluster` + `ld.shared::cluster` to actually
share data; this example tests the dispatch + barrier mechanism alone.

## Run
```
python examples/cluster_basic/run.py
```

## Tutorial
docs/tutorial/19-cluster-cga-intro.md
