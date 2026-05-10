# 64 · Memory Pool Multi-Stream — Cross-Stream Reuse via synchronize_stream

The stream-ordered contract: a block freed on stream A is **only** reusable by
another stream after a synchronization point. CUDA expresses this implicitly
via stream events; Phase 16 makes the synchronization explicit with
`pool.synchronize_stream(stream)`. This keeps the simulator honest about
when cross-stream reuse is and isn't safe.

## What the example does

```python
pool = MemoryPool()
sA, sB = Stream(), Stream()

a = sA.malloc_async(pool, 256)
sA.free_async(pool, a)
sB.malloc_async(pool, 256)            # cannot reach sA's free list — pool grows

# Re-run with explicit synchronize_stream:
pool2 = MemoryPool(); sC, sD = Stream(), Stream()
a2 = sC.malloc_async(pool2, 256)
sC.free_async(pool2, a2)
pool2.synchronize_stream(sC)          # promote sC's free list to cross-stream pool
sD.malloc_async(pool2, 256)           # reuses the promoted block; no growth
```

## 看模拟器

`free_async` pushes the block onto `free_blocks_by_stream[sid]` (per-stream
list). `malloc_async` checks the per-stream list first, then the cross-stream
pool keyed under `-1`.

`synchronize_stream(stream)` pops the entire per-stream list for `sid` and
extends the cross-stream pool with it. Blocks remain pool-owned; ownership of
the slab does not change.

The contract: cross-stream reuse is **only** possible after `synchronize_stream`.
Without it, the cross-stream pool is empty for that block, so the allocator
falls back to `_grow`. This keeps the simulator from silently allowing reuse
that real CUDA would forbid.

## 改一改

- Don't call `synchronize_stream` and observe `len(pool.slabs)` grow on the
  second stream's malloc.
- Synchronize before freeing — `synchronize_stream` is a no-op when nothing is
  on the per-stream list.
- Free on stream A, malloc the same size on stream A *first*, then on stream B
  — same-stream reuse takes priority and B falls through to grow.

## 真机对照

CUDA's stream-ordered allocator detects cross-stream reuse opportunities
through event sync — `cudaEventRecord(event, A)` followed by
`cudaStreamWaitEvent(B, event)` makes B's allocations after the wait eligible
to reuse blocks freed on A before the record. The `cudaMemPoolReuseAllowInternalDependencies`
attribute controls this. Phase 16 keeps the synchronization point explicit
(`synchronize_stream`) — Phase 17 may auto-promote on `record/wait` pairs.
