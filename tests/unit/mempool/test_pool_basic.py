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
