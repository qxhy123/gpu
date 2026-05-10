def test_trim_releases_fully_free_slab_above_threshold():
    """If 2 slabs are fully free and total exceeds threshold, slabs are released."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()

    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a1)
    pool.free_async(s, a2)
    # 2 slabs (1000 + 2000 = 3000 bytes), all free.
    # release_threshold=500 → keep 500 bytes minimum after release.
    released = pool.trim_to(release_threshold_bytes=500)
    # Slabs are released individually if total - this_slab >= 500.
    # Releasing 1000 leaves 2000 ≥ 500 → ok. Releasing 2000 leaves 1000 ≥ 500 → ok.
    # Both released.
    assert released == 3000
    assert all(s is None for s in pool.slabs)


def test_trim_keeps_slab_when_release_would_drop_below_threshold():
    """Release threshold prevents dropping below the floor."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()

    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a1)
    pool.free_async(s, a2)
    # threshold = 2500: releasing the 1000 leaves 2000 < 2500 → keep 1000;
    # releasing the 2000 leaves 1000 < 2500 → keep 2000. Neither released.
    released = pool.trim_to(release_threshold_bytes=2500)
    assert released == 0
    assert all(s is not None for s in pool.slabs)


def test_trim_skips_slabs_with_in_use_bytes():
    """A slab with at least one live allocation cannot be released."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()

    a1 = pool.malloc_async(s, 1000)
    a2 = pool.malloc_async(s, 2000)
    pool.free_async(s, a2)        # only slab 1 is fully free
    released = pool.trim_to(release_threshold_bytes=0)
    # Only slab 1 (2000 bytes) released; slab 0 has a1 in use.
    assert released == 2000
    assert pool.slabs[0] is not None
    assert pool.slabs[1] is None


def test_trim_emits_pool_trim_event():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    a = pool.malloc_async(s, 500)
    pool.free_async(s, a)
    pool.trim_to(release_threshold_bytes=0)
    assert len(rec.pool_trim_events) == 1
    assert rec.pool_trim_events[0].n_bytes_released == 500


def test_trim_no_event_when_nothing_released():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()
    pool.malloc_async(s, 500)        # in-use; cannot release
    pool.trim_to(release_threshold_bytes=0)
    assert len(rec.pool_trim_events) == 0


def test_trim_removes_freed_blocks_pointing_to_released_slabs():
    """Free list must not contain dangling references after trim."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 500)
    pool.free_async(s, a)
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 1
    pool.trim_to(release_threshold_bytes=0)
    # Slab released → freed block pointing to it should also be gone.
    assert len(pool.free_blocks_by_stream[s.stream_id]) == 0
