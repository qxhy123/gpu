import numpy as np
from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream


def main():
    pool = MemoryPool()
    s = Stream()
    print(f"pool {pool._pool_id} created, release_threshold={pool.release_threshold}")

    # 4 alloc/free cycles of 1024 bytes
    for i in range(4):
        a = pool.malloc_async(s, 1024)
        a.buf[:] = i + 1                # touch the buffer
        pool.free_async(s, a)
        print(f"iter {i}: in_flight={pool.in_flight_bytes}, "
                f"high_water={pool.high_water_mark}, slabs={len(pool.slabs)}")

    print(f"final high_water_mark = {pool.high_water_mark} bytes "
            f"(expected 1024 — one slab reused 4x)")


if __name__ == "__main__":
    main()
