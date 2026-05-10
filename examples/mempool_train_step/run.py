import numpy as np
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream
from gpusim.trace.recorder import Recorder
from gpusim.analysis.metrics import (
    pool_high_water_mark, pool_reuse_rate, pool_alloc_count,
)


def main():
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()

    iters = 10
    for i in range(iters):
        act = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
        grad = s.malloc_async(pool, 8 * 1024, dtype=np.float32)
        # Touch buffers (would be a real kernel in production)
        act.buf[:] = float(i)
        grad.buf[:] = float(-i)
        s.free_async(pool, act)
        s.free_async(pool, grad)
        pool.synchronize_stream(s)

    pid = pool._pool_id
    print(f"after {iters} train steps:")
    print(f"  high_water_mark = {pool_high_water_mark(rec, pid)} bytes "
            f"(expected {16 * 1024})")
    print(f"  reuse_rate      = {pool_reuse_rate(rec, pid):.2f} "
            f"(expected {18/20:.2f})")
    print(f"  alloc_count     = {pool_alloc_count(rec, pid)}")
    print(f"  slabs           = {len(pool.slabs)} "
            f"(expected 2 — first iter creates both, rest reuse)")


if __name__ == "__main__":
    main()
