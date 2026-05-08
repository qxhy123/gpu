import numpy as np


def test_tma_desc_then_bulk_copy_into_smem():
    """Issue gpusim.tma_desc to allocate a descriptor, then cp.async.bulk.tensor.2d
    copies a 8x4 fp32 matrix from gmem to smem."""
    import gpusim
    src = """
.entry test(.param .u64 A)
{
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [A];
    // Initialize mbarrier (use smem byte offset 0)
    mov.u64 %rd1, 0;
    mbarrier.init.shared::cta [%rd1], 1;
    // Build TMA descriptor: 4 cols x 8 rows of fp32 (4 bytes each)
    gpusim.tma_desc %rd2, %rd0, 4, 8, 4, 4;
    // Issue bulk copy: smem dst at offset 16, descriptor handle, mbar at offset 0
    mov.u64 %rd3, 16;
    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        [%rd3], [%rd2], [%rd1];
WAIT_LOOP:
    mbarrier.try_wait.parity.shared::cta %p0, [%rd1], 0;
    @!%p0 bra WAIT_LOOP;
    ret;
}
"""
    A = np.arange(32, dtype=np.float32).copy()
    res = gpusim.run(ptx_src=src, grid=(1, 1, 1), block=(32, 1, 1),
                     params={"A": A}, mode="timing")
    assert res.metrics["cycles"] > 0
    assert res.metrics["cycles"] < 1000
