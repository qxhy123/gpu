def test_allocation_buf_is_writable_typed_view():
    """Pool buffer is a numpy view that supports element write + readback."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 16, dtype=np.uint32)
    assert a.buf.shape == (4,)            # 16 bytes / 4 = 4 elements
    a.buf[:] = [1, 2, 3, 4]
    assert list(a.buf) == [1, 2, 3, 4]


def test_allocation_buf_writes_persist_through_free_and_realloc():
    """Pool does not zero on free; readers see prior writer's data until they overwrite."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a1 = pool.malloc_async(s, 16, dtype=np.uint8)
    a1.buf[:] = 0x55
    pool.free_async(s, a1)
    a2 = pool.malloc_async(s, 16, dtype=np.uint8)
    assert (a2.buf == 0x55).all()


def test_allocation_buf_works_as_kernel_param():
    """Allocation.buf is shape-compatible with kernel param expectations
    (numpy ndarray with .shape, .dtype, .data, indexing). We don't actually launch
    a kernel here — gpusim's launcher reads ndarrays via numpy attributes only."""
    import numpy as np
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    pool = MemoryPool()
    s = Stream()
    a = pool.malloc_async(s, 32, dtype=np.uint32)
    # Ensure ndarray protocol
    assert hasattr(a.buf, "shape")
    assert hasattr(a.buf, "dtype")
    assert hasattr(a.buf, "data")
    assert isinstance(a.buf, np.ndarray)
    # Indexable
    a.buf[0] = 42
    assert int(a.buf[0]) == 42
