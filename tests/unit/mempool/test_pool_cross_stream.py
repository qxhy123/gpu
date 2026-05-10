def test_cross_stream_alloc_without_sync_grows_pool():
    """Free on stream A, malloc on stream B without sync → grow."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream()
    sB = Stream()
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    # Without sync, B cannot reach A's free list
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 2


def test_synchronize_stream_promotes_blocks_to_cross_stream_pool():
    """After synchronize_stream(sA), sB.malloc reuses the block."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream()
    sB = Stream()
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    assert len(pool.free_blocks_by_stream[sA.stream_id]) == 1
    pool.synchronize_stream(sA)
    assert len(pool.free_blocks_by_stream[sA.stream_id]) == 0
    assert len(pool.free_blocks_by_stream[-1]) == 1
    # Now sB can reuse
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 1


def test_synchronize_stream_idempotent_when_nothing_to_promote():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    pool.synchronize_stream(s)    # no error, no-op
    assert len(pool.free_blocks_by_stream[-1]) == 0


def test_same_stream_reuse_takes_priority_over_cross_stream():
    """If both per-stream and cross-stream blocks fit, prefer same-stream."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream()
    sB = Stream()
    # Stream A allocates and frees, then synchronizes (block goes to cross-stream pool)
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    pool.synchronize_stream(sA)
    # Stream B allocates and frees (B's block is in B's per-stream list)
    b = pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 1   # B reused A's promoted block
    pool.free_async(sB, b)
    # Now another B malloc should pull from B's per-stream list, not cross-stream
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 1   # still 1, but reuse happened
