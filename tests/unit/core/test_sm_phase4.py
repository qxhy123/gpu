def test_sm_accepts_external_l2_hbm():
    from gpusim.config.loader import load_default
    from gpusim.core.sm import SM
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.core.hbm import HBM
    cfg = load_default()
    hbm = HBM(cfg.hbm, recorder=None)
    l2 = L2Cache(cfg.cache, hbm, recorder=None)
    sm = SM(cfg.sm, sm_id=3, recorder=None, l2=l2, hbm=hbm)
    assert sm.sm_id == 3
    assert sm.l2 is l2
    assert sm.hbm is hbm


def test_sm_can_admit_cta_returns_bool():
    from gpusim.config.loader import load_default
    from gpusim.core.sm import SM
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.core.hbm import HBM
    cfg = load_default()
    hbm = HBM(cfg.hbm); l2 = L2Cache(cfg.cache, hbm)
    sm = SM(cfg.sm, sm_id=0, recorder=None, l2=l2, hbm=hbm)
    class _Occ:
        active_ctas = 32
    assert sm.can_admit_cta(_Occ()) is True


def test_sm_active_warp_count_zero_initially():
    from gpusim.config.loader import load_default
    from gpusim.core.sm import SM
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.core.hbm import HBM
    cfg = load_default()
    hbm = HBM(cfg.hbm); l2 = L2Cache(cfg.cache, hbm)
    sm = SM(cfg.sm, sm_id=0, recorder=None, l2=l2, hbm=hbm)
    assert sm.active_warp_count() == 0
