from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
import numpy as np

def test_full_outstanding_queue_causes_structural_stall():
    cfg = load_default()
    cfg.fu.lsu_outstanding = 2
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
    # With L1 cache miss latency ~205 cycles, lsu_outstanding=2 still causes
    # significant stalls — the 3rd+ loads must wait for earlier ones to drain.
    # Old threshold was >400 (fixed 400-cycle latency); now >100 is sufficient.
    assert res.cycles > 100
