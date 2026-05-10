# mempool_multi_stream — Phase 16

Two streams sharing a pool. Demonstrates that cross-stream reuse requires
explicit `pool.synchronize_stream(stream)` to promote per-stream free blocks
into the cross-stream pool.

## Run
```bash
python run.py
```
