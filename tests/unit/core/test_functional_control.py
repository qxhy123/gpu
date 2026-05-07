import numpy as np
from gpusim.core.exec import functional_run
from gpusim.frontend.parser import parse

def test_branch_divergence_writes_per_lane():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<6>; .reg .u64 %rd<3>; .reg .pred %p<2>; .reg .f32 %f<2>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        cvt.u64.u32 %rd2, %r1;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd3, %r2;
        add.u64 %rd2, %rd1, %rd3;
        setp.lt.s32 %p1, %r1, 16;
        @%p1 bra THEN;
        mov.u32 %r3, 100;
        bra DONE;
        THEN:
        mov.u32 %r3, 200;
        DONE:
        st.global.u32 [%rd2], %r3;
    }
    """
    out = np.zeros(32, dtype=np.uint32)
    functional_run(src, params={"OUT": out}, grid=(1,1,1), block=(32,1,1))
    expected = np.array([200]*16 + [100]*16, dtype=np.uint32)
    np.testing.assert_array_equal(out, expected)
