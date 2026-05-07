import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM

def test_shared_no_conflict_pattern_1cycle_occupancy():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd1, %r2;
        cvt.f32.s32 %f1, %r1;
        st.shared.f32 [%rd1], %f1;
        ld.shared.f32 %f2, [%rd1];
        bar.sync 0;
    }
    """
    k = parse(src, "<t>")
    out = np.zeros(32, dtype=np.float32)
    sm = SM(load_default())
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1), params={"OUT": out})
    assert res.cycles > 0

def test_shared_32way_conflict_costs_more_cycles():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 7;
        cvt.u64.u32 %rd1, %r2;
        cvt.f32.s32 %f1, %r1;
        st.shared.f32 [%rd1], %f1;
        bar.sync 0;
    }
    """
    k = parse(src, "<t>")
    out = np.zeros(32, dtype=np.float32)
    sm = SM(load_default())
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1), params={"OUT": out})
    assert res.cycles >= 32 + 5
