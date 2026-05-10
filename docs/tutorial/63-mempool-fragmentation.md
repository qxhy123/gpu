# 63 · Memory Pool Fragmentation — Best-Fit + trim_to

When allocations of mixed sizes are freed and re-requested, the pool's
**best-fit** policy picks the smallest free block that fits. Among equal-size
blocks, the **oldest free** wins (FIFO tiebreak).

This chapter walks an example that:
1. Allocates 5 mixed-size blocks: `[1024, 2048, 4096, 1024, 2048]`.
2. Frees all 5.
3. Re-allocates 1024 + 2048 — best-fit picks the right blocks; pool does
   **not** grow.
4. Frees the new allocations and calls `pool.trim_to(0)` — all 5 fully-free
   slabs are released back to the OS.

## What the example does

```python
pool = MemoryPool(); s = Stream()
allocs = [pool.malloc_async(s, n) for n in (1024, 2048, 4096, 1024, 2048)]
for a in allocs: pool.free_async(s, a)
a1 = pool.malloc_async(s, 1024)        # reuses oldest 1024 (slot 0)
a2 = pool.malloc_async(s, 2048)        # reuses oldest 2048 (slot 1)
print(len(pool.slabs))                  # 5 — no new growth

for a in (a1, a2): pool.free_async(s, a)
released = pool.trim_to(release_threshold_bytes=0)
print(released)                         # full size of all 5 slabs
```

## 看模拟器

`_pop_best_fit` walks the candidate list, sorts by `(n_bytes, freed_at_count)`,
and pops the smallest-fit-then-oldest-free block. Returning the original block
size (no splitting) means a 50-byte request from a 64-byte free block yields
a 64-byte allocation — splitting is YAGNI for now.

`trim_to(release_threshold_bytes)` aggregates per-slab free bytes; only slabs
with `per_slab_free == slab_size` are eligible. It releases each eligible slab
individually as long as `current_total - this_slab >= release_threshold_bytes`.
After release, slabs in `pool.slabs` become `None` (preserving indices) and
free blocks pointing to them are pruned from `free_blocks_by_stream`.

## 改一改

- Set `release_threshold_bytes` higher than total slab bytes — `trim_to`
  releases nothing; `pool.slabs` stays untouched.
- Hold one allocation back unfreed before calling `trim_to`. That slab is not
  fully free, so it survives.
- Free in a different order to see oldest-free tiebreak in action.

## 真机对照

CUDA: `cudaMemPoolTrimTo(pool, minBytesToKeep)` releases unused memory above
the threshold. The matching attribute is
`cudaMemPoolAttrReleaseThreshold`. PyTorch exposes this as
`torch.cuda.empty_cache()`; JAX exposes it through XLA's allocator. The
release-threshold semantics in Phase 16 mirror the CUDA contract — keep the
pool above the floor, drop slabs above it.
