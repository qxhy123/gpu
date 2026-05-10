def test_stream_malloc_async_forwards_to_pool():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = s.malloc_async(pool, 32)
    assert a.n_bytes == 32
    assert a.alloc_stream_id == s.stream_id


def test_stream_free_async_forwards_to_pool():
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = s.malloc_async(pool, 32)
    assert pool.in_flight_bytes == 32
    s.free_async(pool, a)
    assert pool.in_flight_bytes == 0


def test_stream_malloc_async_dtype_arg_passthrough():
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = s.malloc_async(pool, 16, dtype=np.float32)
    assert a.buf.dtype == np.float32
    assert a.buf.shape == (4,)    # 16 / 4 = 4 elements
