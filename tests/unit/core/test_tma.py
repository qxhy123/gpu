import numpy as np


def test_descriptor_pool_allocate_and_lookup():
    from gpusim.core.tma import TensorDescriptorPool
    pool = TensorDescriptorPool()
    handle = pool.allocate(gmem_base=0x10000000, dim_x=128, dim_y=64,
                            stride_y=512, elem_bytes=2)
    assert handle == 0
    desc = pool.lookup(handle)
    assert desc.gmem_base == 0x10000000 and desc.dim_x == 128


def test_bulk_copy_2d_copies_correct_bytes():
    """cp.async.bulk.tensor.2d.global.shared performs gmem→smem 2D copy."""
    from gpusim.core.exec import GlobalMemory, SharedMemory
    from gpusim.core.tma import TensorDescriptorPool, do_bulk_copy_2d

    g = GlobalMemory()
    src_arr = np.arange(64 * 32, dtype=np.float16).reshape(64, 32)
    src_base = g.bind("A", src_arr.flatten().copy())

    s = SharedMemory(size_bytes=8192)
    s.allocate_cta(0, 8192)

    pool = TensorDescriptorPool()
    handle = pool.allocate(gmem_base=src_base, dim_x=32, dim_y=64,
                            stride_y=32, elem_bytes=2)
    desc = pool.lookup(handle)
    smem_dst = 0
    do_bulk_copy_2d(gmem=g, smem=s, cta_id=0, smem_dst=smem_dst, desc=desc)
    n = 64 * 32 * 2
    expected = src_arr.flatten().tobytes()
    actual = bytes(s._cta[0][smem_dst:smem_dst + n])
    assert actual == expected
