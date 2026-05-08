from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
import numpy as np

def test_full_outstanding_queue_causes_structural_stall():
    dev_cfg = load_default()
    cfg = dev_cfg.sm
    cfg.fu.lsu_outstanding = 2
    cfg._cache_for_run = dev_cfg.cache
    cfg._hbm_for_run = dev_cfg.hbm
    src = """
    .visible .entry k(.param .u64 A) {
        .reg .u32 %r<6>; .reg .u64 %rd<6>; .reg .f32 %f<8>;
        ld.param.u64 %rd1, [A];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        ld.global.f32 %f1, [%rd3];
        ld.global.f32 %f2, [%rd3];
        ld.global.f32 %f3, [%rd3];
        ld.global.f32 %f4, [%rd3];
        ld.global.f32 %f5, [%rd3];
        bar.sync 0;
    }
    """
    arr = np.arange(32, dtype=np.float32)
    k = parse(src, "<t>")
    sm = SM(cfg)
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1), params={"A": arr})
    # With real HBM row_miss_latency=30 (+L2 install 10 + L1 miss check 5 = 45 cy),
    # lsu_outstanding=2 forces the 3rd+ loads to stall until earlier ones drain.
    # Without limit: ~21 cycles; with limit=2: ~64 cycles.
    # Threshold >30 is safely above the no-limit baseline.
    assert res.cycles > 30
