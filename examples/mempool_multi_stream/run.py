from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream


def main():
    # Without synchronize_stream
    pool_a = MemoryPool()
    sA = Stream(); sB = Stream()
    a = sA.malloc_async(pool_a, 256)
    sA.free_async(pool_a, a)
    sB.malloc_async(pool_a, 256)    # cannot reuse — grows
    print(f"unsynced: {len(pool_a.slabs)} slabs (expected 2)")

    # With synchronize_stream
    pool_b = MemoryPool()
    sC = Stream(); sD = Stream()
    a2 = sC.malloc_async(pool_b, 256)
    sC.free_async(pool_b, a2)
    pool_b.synchronize_stream(sC)
    sD.malloc_async(pool_b, 256)    # reuses
    print(f"synced: {len(pool_b.slabs)} slabs (expected 1)")


if __name__ == "__main__":
    main()
