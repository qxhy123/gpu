# 62 · Memory Pool Basic — Stream-Ordered alloc/free with Reuse

Phase 16 introduces `gpusim.mempool.MemoryPool` — a stream-ordered allocator
modeled after CUDA's `cudaMallocAsync` / `cudaFreeAsync` family. Each
allocation returns an `Allocation` whose `.buf` attribute is a typed numpy
view, so kernels accept it just like any other buffer.

This chapter walks the simplest pattern: one stream allocates and frees a
1 KB block four times. The first allocation grows the pool; the next three
reuse the freed block.

## What the example does

```python
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream

pool = MemoryPool()
s = Stream()
for i in range(4):
    a = pool.malloc_async(s, 1024)
    a.buf[:] = i + 1                # write into the buffer
    pool.free_async(s, a)
print(pool.high_water_mark, len(pool.slabs))   # 1024 1
```

## 看模拟器

`malloc_async` first checks the per-stream free list (`free_blocks_by_stream[s.stream_id]`).
On the first iteration, the list is empty so `_grow` allocates a fresh
`bytearray(1024)` slab and returns a numpy view via `np.frombuffer`.

`free_async` pushes a `_FreeBlock(n_bytes=1024, slab_index=0, byte_offset=0,
freed_at_count=N)` onto the per-stream free list and decrements
`in_flight_bytes`.

On iteration 2, `malloc_async` finds the free block, pops it, and reuses the
same slab — no growth. `pool.slabs` stays length 1, `high_water_mark` stays
at 1024.

The recorder (when attached as `pool._recorder = rec`) emits a `PoolAllocate`
event each time, with `reused=False` for the first call and `reused=True`
afterward. The metric `pool_reuse_rate(recorder, pool_id)` reports the
fraction of `PoolAllocate.reused == True` events.

## 改一改

- Allocate two blocks of 1024 each before freeing. Now `pool.slabs == 2`,
  `high_water_mark == 2048`, and the next iter reuses the most recently freed
  block (oldest-free tiebreak).
- Pass `dtype=np.float32`: the `buf` view is shape `(256,)`, dtype `float32`,
  and reads/writes go through the same underlying `bytearray`.

## 真机对照

Real CUDA: `cudaMallocAsync(&ptr, n, stream)` and `cudaFreeAsync(ptr, stream)`.
Free on stream A makes the block reusable by subsequent allocations on stream A
immediately. PyTorch's caching allocator wraps these and adds heuristics for
block splitting and rounding (we keep best-fit unsplit). Phase 17 may add
splitting and the `cudaMemPoolReuseAllowOpportunistic` semantics.
