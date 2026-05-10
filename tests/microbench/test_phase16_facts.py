"""Phase 16 microbench — memory pool facts."""


def test_first_alloc_is_fresh_second_after_free_is_reused():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    a1 = pool.malloc_async(s, 64)
    pool.free_async(s, a1)
    pool.malloc_async(s, 64)
    assert rec.pool_allocate_events[0].reused is False
    assert rec.pool_allocate_events[1].reused is True


def test_cross_stream_alloc_without_sync_grows():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    sA = Stream(); sB = Stream()
    a = pool.malloc_async(sA, 64)
    pool.free_async(sA, a)
    pool.malloc_async(sB, 64)
    assert len(pool.slabs) == 2


def test_trim_respects_release_threshold():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a1)
    pool.free_async(s, a2)
    # threshold 2500: neither slab can be released without dropping below.
    released = pool.trim_to(release_threshold_bytes=2500)
    assert released == 0


def test_high_water_mark_monotone_non_decreasing():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    history = []
    a1 = pool.malloc_async(s, 100); history.append(pool.high_water_mark)
    a2 = pool.malloc_async(s, 200); history.append(pool.high_water_mark)
    pool.free_async(s, a1);          history.append(pool.high_water_mark)
    pool.free_async(s, a2);          history.append(pool.high_water_mark)
    assert history == [100, 300, 300, 300]


def test_pool_id_increments():
    from gpusim.mempool.pool import MemoryPool
    p1 = MemoryPool()
    p2 = MemoryPool()
    p3 = MemoryPool()
    assert p2._pool_id == p1._pool_id + 1
    assert p3._pool_id == p2._pool_id + 1


def test_synchronize_stream_promotes_blocks():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 64)
    pool.free_async(s, a)
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 1
    pool.synchronize_stream(s)
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 0
    assert len(pool.free_blocks_by_stream[-1]) == 1
