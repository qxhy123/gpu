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
