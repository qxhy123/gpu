# phase8_overlap_real

Phase 9 demo: same compute+memory kernels as Phase 8 true_concurrent_overlap,
this time with the per-cycle main loop. Demonstrates `total_cycles <= sum-of-per-launch`
overlap awareness via the `cross_stream_concurrency_gain()` metric.

Note: Phase 9 M1 minimal -- true per-cycle CTA interleave (vs M1's per-launch
processing) requires Device.run cycle-slicing which is deferred.

## Run
```
python examples/phase8_overlap_real/run.py
```

## Tutorial
docs/tutorial/37-per-cycle-scheduler-and-real-overlap.md
