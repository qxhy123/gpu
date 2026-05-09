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
