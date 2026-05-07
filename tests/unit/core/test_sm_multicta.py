import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
from gpusim.core.occupancy import compute_occupancy

def test_multi_cta_runs_concurrently_faster_than_serial():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<5>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        mov.u32 %r2, %ctaid.x;
        mad.lo.s32 %r3, %r2, 32, %r1;
        shl.b32 %r4, %r3, 2;
        cvt.u64.u32 %rd2, %r4;
        add.u64 %rd1, %rd1, %rd2;
        st.global.u32 [%rd1], %r3;
        bar.sync 0;
    }
    """
    out = np.zeros(128, dtype=np.uint32)
    cfg = load_default()
    sm = SM(cfg)
    res = sm.run(kernel=parse(src, "<t>"), grid=(4,1,1), block=(32,1,1),
                 params={"OUT": out})
    res1 = sm.run(kernel=parse(src, "<t>"), grid=(1,1,1), block=(32,1,1),
                  params={"OUT": np.zeros(32, dtype=np.uint32)})
    assert res.cycles < 3 * res1.cycles
    assert list(out) == list(range(128))
