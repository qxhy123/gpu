def test_best_fit_among_multiple_candidates_picks_smallest():
    """3 free blocks >= requested → smallest is picked."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a64 = pool.malloc_async(s, 64)
    a128 = pool.malloc_async(s, 128)
    a256 = pool.malloc_async(s, 256)
    pool.free_async(s, a256)
    pool.free_async(s, a128)
    pool.free_async(s, a64)
    a_new = pool.malloc_async(s, 50)    # smallest fit is a64
    assert a_new._slab_index == a64._slab_index


def test_best_fit_tiebreak_oldest_free_when_sizes_equal():
    """Two equal-size free blocks → oldest free wins."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 64)
    a2 = pool.malloc_async(s, 64)
    pool.free_async(s, a1)               # freed first (oldest)
    pool.free_async(s, a2)               # freed second
    a_new = pool.malloc_async(s, 64)
    # a1 was freed first → its block should be reused
    assert a_new._slab_index == a1._slab_index


def test_best_fit_block_keeps_full_size_when_oversized():
    """Requesting 50 from a 64-byte block returns a 64-byte allocation (no splitting)."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 64)
    pool.free_async(s, a)
    a_new = pool.malloc_async(s, 50)
    assert a_new.n_bytes == 64    # block size, not request size
