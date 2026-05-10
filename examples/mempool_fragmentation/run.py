from gpusim.mempool.pool import MemoryPool
from gpusim.api import Stream


def main():
    pool = MemoryPool()
    s = Stream()
    sizes = [1024, 2048, 4096, 1024, 2048]
    allocs = [pool.malloc_async(s, n) for n in sizes]
    print(f"after 5 alloc: {len(pool.slabs)} slabs (expected 5)")

    for a in allocs:
        pool.free_async(s, a)

    a1 = pool.malloc_async(s, 1024)
    a2 = pool.malloc_async(s, 2048)
    print(f"after re-alloc 1024+2048: {len(pool.slabs)} slabs (expected 5, no growth)")
    print(f"reused 1024 from slab {a1._slab_index}, 2048 from slab {a2._slab_index}")

    # Show trim behavior
    for a in (a1, a2):
        pool.free_async(s, a)
    released = pool.trim_to(release_threshold_bytes=0)
    print(f"trim_to(0) released {released} bytes (all 5 slabs)")


if __name__ == "__main__":
    main()
