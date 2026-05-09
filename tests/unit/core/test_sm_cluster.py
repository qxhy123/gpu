def test_sm_activate_cta_propagates_cluster_id_to_warps():
    from gpusim.config.loader import load_default
    from gpusim.core.sm import SM
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.core.hbm import HBM
    from gpusim.core.exec import GlobalMemory, SharedMemory, ParamSpace
    from gpusim.core.occupancy import compute_occupancy

    cfg = load_default()
    cfg.cluster_size = 4
    hbm = HBM(cfg.hbm); l2 = L2Cache(cfg.cache, hbm)
    sm = SM(cfg.sm, sm_id=0, recorder=None, l2=l2, hbm=hbm)

    from gpusim.frontend.parser import parse
    k = parse(".entry test() { ret; }", "<test>")
    gmem = GlobalMemory(); smem = SharedMemory()
    ps = ParamSpace({})
    occ = compute_occupancy(cfg.sm, threads_per_cta=32, regs_per_thread=16, smem_per_cta=0)
    sm.initialize_for_run(kernel=k, gmem=gmem, smem=smem, paramspace=ps,
                            grid=(4,1,1), block=(32,1,1), occupancy=occ,
                            cluster_size=4)
    sm.activate_cta(cta_id=2, ctaid_xyz=(2,0,0), regs_per_thread=16,
                     smem_per_cta=0, threads_per_cta=32, warps_per_cta=1,
                     cycle=0, cluster_id=0, cluster_rank=2)
    assert sm.active_warp_count() == 1
    w = sm._active_warps[0]
    assert w.cluster_id == 0
    assert w.cluster_rank == 2


def test_sm_activate_cta_default_no_cluster():
    """Without cluster_id/rank kwargs, defaults to -1 (no cluster)."""
    from gpusim.config.loader import load_default
    from gpusim.core.sm import SM
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.core.hbm import HBM
    from gpusim.core.exec import GlobalMemory, SharedMemory, ParamSpace
    from gpusim.core.occupancy import compute_occupancy
    cfg = load_default()
    hbm = HBM(cfg.hbm); l2 = L2Cache(cfg.cache, hbm)
    sm = SM(cfg.sm, sm_id=0, recorder=None, l2=l2, hbm=hbm)
    from gpusim.frontend.parser import parse
    k = parse(".entry test() { ret; }", "<test>")
    gmem = GlobalMemory(); smem = SharedMemory()
    ps = ParamSpace({})
    occ = compute_occupancy(cfg.sm, threads_per_cta=32, regs_per_thread=16, smem_per_cta=0)
    sm.initialize_for_run(kernel=k, gmem=gmem, smem=smem, paramspace=ps,
                            grid=(1,1,1), block=(32,1,1), occupancy=occ)
    sm.activate_cta(cta_id=0, ctaid_xyz=(0,0,0), regs_per_thread=16,
                     smem_per_cta=0, threads_per_cta=32, warps_per_cta=1,
                     cycle=0)
    w = sm._active_warps[0]
    assert w.cluster_id == -1
    assert w.cluster_rank == -1


def test_cluster_barrier_arrive_wait_synchronizes_2_ctas():
    """2 CTAs in cluster_size=2 cluster: each CTA writes its rank to OUT[rank];
    both go through barrier.cluster.{arrive,wait} without deadlock."""
    import numpy as np
    from gpusim.config.schema import DeviceConfig
    from gpusim.core.device import Device
    from gpusim.frontend.parser import parse
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<3>;
    .reg .u32 %r<3>;
    .reg .pred %p0;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %ctaid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, %tid.x;
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra END;
    st.global.u32 [%rd2], %r0;
    barrier.cluster.arrive;
    barrier.cluster.wait;
END:
    ret;
}
"""
    cfg = DeviceConfig(n_sm=2, cluster_size=2)
    out = np.zeros(2, dtype=np.uint32)
    dev = Device(cfg)
    res = dev.run(kernel=parse(src, "<test>"), grid=(2,1,1), block=(32,1,1),
                   params={"OUT": out})
    assert (out == np.array([0, 1], dtype=np.uint32)).all()
    assert res.cycles > 0
    assert res.cycles < 10_000
