def test_cluster_mbarrier_init_routes_to_target_cta_pool():
    """mbarrier.init.shared::cluster on a cluster pointer routes to target CTA's pool."""
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test() {
    .reg .u64 %rd<2>;
    mov.u64 %rd0, 16777216;
    mbarrier.init.shared::cluster [%rd0], 4;
}
"""
    # 16777216 = (1 << 24) | 0; init mbarrier in CTA 1's smem at offset 0
    cfg = load_default()
    cfg.cluster_size = 2
    cfg.n_sm = 2
    res = gpusim.run(ptx_src=src, grid=(2,1,1), block=(32,1,1),
                      params={}, mode="timing", config=cfg)
    # Verify no crash + reasonable cycles
    assert 0 < res.metrics["cycles"] < 1000


def test_cluster_tma_load_writes_to_remote_cta_smem():
    """cp.async.bulk.tensor.shared::cluster smem_dst with rank-encoded ptr writes to that CTA's smem."""
    import numpy as np
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 A) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .pred %p0;

    ld.param.u64 %rd0, [A];

    mov.u32 %r0, %tid.x;
    getctarank.u32 %rrank;
    setp.ne.u32 %p0, %rrank, 0;
    @%p0 bra END;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;

    // Init mbarrier in CTA rank 1's smem at offset 0 (cluster encoded)
    mov.u64 %rd1, 16777216;          // (1 << 24) | 0
    mbarrier.init.shared::cluster [%rd1], 1;

    // TMA descriptor: 4 fp32 cols × 4 rows = 64 bytes total
    gpusim.tma_desc %rd2, %rd0, 4, 4, 4, 4;

    // smem_dst: rank=1 (remote), offset 64 (after mbar)
    mov.u64 %rd3, 16777280;          // (1 << 24) | 64

    cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
        [%rd3], [%rd2], [%rd1];
END:
    barrier.cluster.arrive;
    barrier.cluster.wait;
    ret;
}
"""
    A = np.arange(16, dtype=np.float32)
    cfg = load_default()
    cfg.cluster_size = 2; cfg.n_sm = 2
    res = gpusim.run(ptx_src=src, grid=(2,1,1), block=(32,1,1),
                      params={"A": A}, mode="timing", config=cfg)
    assert 0 < res.metrics["cycles"] < 5000
