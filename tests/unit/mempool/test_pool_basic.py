def test_allocation_holds_buf_and_metadata():
    from gpusim.mempool.allocation import Allocation
    import numpy as np
    buf = np.zeros(8, dtype=np.uint8)
    a = Allocation(ptr_id=1, n_bytes=8, buf=buf, pool=None,
                    alloc_stream_id=3, _slab_index=0, _byte_offset=0)
    assert a.ptr_id == 1
    assert a.n_bytes == 8
    assert a.buf is buf
    assert a.alloc_stream_id == 3
    assert a._slab_index == 0
    assert a._byte_offset == 0


def test_pool_first_malloc_grows():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 64)
    assert a.n_bytes == 64
    assert a.alloc_stream_id == s.stream_id
    assert pool.in_flight_bytes == 64
    assert pool.high_water_mark == 64
    assert len(pool.slabs) == 1


def test_pool_second_alloc_after_free_reuses_same_stream():
    """Free on stream A → next malloc on A immediately reuses."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 64)
    pool.free_async(s, a1)
    assert pool.in_flight_bytes == 0
    a2 = pool.malloc_async(s, 64)
    # Same slab — no new growth
    assert len(pool.slabs) == 1
    assert a2._slab_index == a1._slab_index
    assert a2._byte_offset == a1._byte_offset


def test_pool_in_flight_decreases_on_free():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 32)
    assert pool.in_flight_bytes == 32
    pool.free_async(s, a)
    assert pool.in_flight_bytes == 0


def test_pool_high_water_mark_monotone():
    """high_water_mark only increases, never decreases."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 100)
    a2 = pool.malloc_async(s, 200)   # in_flight = 300
    assert pool.high_water_mark == 300
    pool.free_async(s, a1)            # in_flight = 200
    assert pool.high_water_mark == 300
    pool.free_async(s, a2)            # in_flight = 0
    assert pool.high_water_mark == 300


def test_pool_id_unique():
    from gpusim.mempool.pool import MemoryPool
    p1 = MemoryPool()
    p2 = MemoryPool()
    assert p1._pool_id != p2._pool_id


def test_allocation_buf_is_numpy_view_zeros_initial():
    """Newly-grown slab is a fresh bytearray; the view starts as zeros."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 16, dtype=np.uint8)
    assert isinstance(a.buf, np.ndarray)
    assert a.buf.dtype == np.uint8
    assert a.buf.shape == (16,)
    assert (a.buf == 0).all()


def test_allocation_buf_writes_back_to_slab():
    """Mutating the view mutates the underlying bytearray; reuse sees writes (until the
    new owner overwrites — they are responsible for initialization)."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 16, dtype=np.uint8)
    a1.buf[:] = 7
    pool.free_async(s, a1)
    a2 = pool.malloc_async(s, 16, dtype=np.uint8)
    # Same memory; not zeroed by the pool — caller's responsibility.
    assert (a2.buf == 7).all()


def test_pool_best_fit_picks_smallest_fitting():
    """Best-fit: among free blocks >= requested, pick the smallest."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a_small = pool.malloc_async(s, 32)
    a_med = pool.malloc_async(s, 64)
    a_large = pool.malloc_async(s, 128)
    # Free all
    pool.free_async(s, a_small)
    pool.free_async(s, a_med)
    pool.free_async(s, a_large)
    # Request 50 — should reuse the 64-byte block, not the 128-byte one
    a_new = pool.malloc_async(s, 50)
    assert a_new._slab_index == a_med._slab_index
    assert a_new._byte_offset == a_med._byte_offset
    assert a_new.n_bytes == 64    # block keeps its full size


def test_pool_best_fit_grows_when_no_block_fits():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a_small = pool.malloc_async(s, 32)
    pool.free_async(s, a_small)
    # Request 100 — no fit, grow
    a_new = pool.malloc_async(s, 100)
    assert len(pool.slabs) == 2
    assert a_new.n_bytes == 100
