def test_cluster_tma_store_routes_smem_src_to_remote_cta():
    """cp.async.bulk.tensor.2d.global.shared::cluster reads from cluster-encoded smem_src."""
    import numpy as np
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<8>;
    .reg .u32 %r<4>;
    .reg .u32 %rrank;
    .reg .pred %p0;

    ld.param.u64 %rd0, [OUT];

    mov.u32 %r0, %tid.x;
    getctarank.u32 %rrank;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;

    // CTA rank 1 fills its smem with values 100..115
    setp.ne.u32 %p0, %rrank, 1;
    @%p0 bra MAYBE_STORE;
    mov.u32 %r1, 100;
    mov.u64 %rd1, 0;
    st.shared.u32 [%rd1], %r1;

MAYBE_STORE:
    barrier.cluster.arrive;
    barrier.cluster.wait;

    // CTA rank 0: TMA store from rank 1's smem to OUT
    setp.ne.u32 %p0, %rrank, 0;
    @%p0 bra END;

    gpusim.tma_desc %rd2, %rd0, 1, 1, 1, 4;
    mov.u64 %rd3, 16777216;        // (1 << 24) | 0 = rank 1, offset 0
    cp.async.bulk.tensor.2d.global.shared::cluster [%rd2], [%rd3];
    cp.async.bulk.commit_group;
    cp.async.bulk.wait_group 0;
END:
    barrier.cluster.arrive;
    barrier.cluster.wait;
    ret;
}
"""
    cfg = load_default()
    cfg.cluster_size = 2; cfg.n_sm = 2
    out = np.zeros(1, dtype=np.uint32)
    res = gpusim.run(ptx_src=src, grid=(2,1,1), block=(32,1,1),
                      params={"OUT": out}, mode="timing", config=cfg)
    assert int(out[0]) == 100
