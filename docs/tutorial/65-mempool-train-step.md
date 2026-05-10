# 65 · Memory Pool Train-Step — Reuse Convergence

A typical training loop allocates buffers (activations, gradients, temporaries)
each iteration and frees them at the end. Without a pool, every iteration would
hit the kernel allocator. With a pool, the first iteration warms up the working
set; every iteration after that reuses the same blocks. The high-water mark
plateaus, and the reuse rate climbs toward 1.

## What the example does

```python
pool = MemoryPool()
pool._recorder = rec
s = Stream()
for _ in range(10):
    act = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
    grad = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
    # ... real kernel would compute on act/grad here ...
    s.free_async(pool, act)
    s.free_async(pool, grad)
    pool.synchronize_stream(s)
# high_water_mark plateaus at 16 KB; reuse_rate = 18/20 = 0.9
```

## 看模拟器

Iteration 1: both `malloc_async` calls miss the free list and `_grow` adds a
slab each. `high_water_mark` rises from 0 → 8 KB → 16 KB.

Iteration 2-10: each `malloc_async` finds the freed block from the previous
iteration on the per-stream list (best-fit, oldest-free wins). Each malloc
emits a `PoolAllocate` event with `reused=True`.

After 10 iterations: 20 total `malloc_async` calls. 2 fresh
(`reused=False`), 18 reused → `pool_reuse_rate = 0.9`. `pool_high_water_mark`
stays at 16 KB. `len(pool.slabs) == 2`.

## 改一改

- Increase the iter count to 100 — `reuse_rate` approaches 1.0
  (`(100 * 2 - 2) / (100 * 2) = 0.99`).
- Add a third allocation per iteration with a varying size. The pool grows
  larger but still plateaus.
- Skip `synchronize_stream` and call free on a different stream from
  malloc — see the pool grow because cross-stream reuse is unavailable.

## 真机对照

PyTorch's caching allocator achieves the same effect transparently —
`tensor.zero_()` in a loop never round-trips to the CUDA allocator after the
first iter. JAX's XLA allocator does the same with even larger working sets.
The metric to watch is `torch.cuda.memory_stats()["allocated_bytes.peak"]`,
which is the equivalent of Phase 16's `pool_high_water_mark`.
