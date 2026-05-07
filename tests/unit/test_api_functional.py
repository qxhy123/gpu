import numpy as np
import gpusim

VECTOR_ADD = """
.visible .entry vec_add(.param .u64 A, .param .u64 B, .param .u64 C, .param .u32 N) {
    .reg .u32 %r<6>; .reg .f32 %f<4>; .reg .u64 %rd<6>; .reg .pred %p<2>;
    ld.param.u64 %rd1, [A];
    ld.param.u64 %rd2, [B];
    ld.param.u64 %rd3, [C];
    ld.param.u32 %r1, [N];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    setp.ge.s32 %p1, %r5, %r1;
    @%p1 bra END;
    shl.b32 %r6, %r5, 2;
    cvt.u64.u32 %rd4, %r6;
    add.u64 %rd5, %rd1, %rd4;
    add.u64 %rd6, %rd2, %rd4;
    ld.global.f32 %f1, [%rd5];
    ld.global.f32 %f2, [%rd6];
    add.f32 %f3, %f1, %f2;
    add.u64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f3;
    END: bar.sync 0;
}
"""

def test_vector_add_functional_parity():
    n = 1024
    a = np.random.RandomState(0).randn(n).astype(np.float32)
    b = np.random.RandomState(1).randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    r = gpusim.run(
        ptx_src=VECTOR_ADD,
        grid=(8,1,1), block=(128,1,1),
        params={"A": a, "B": b, "C": c, "N": n},
        mode="functional",
    )
    np.testing.assert_allclose(c, a + b, rtol=1e-5)
    assert r.outputs["C"] is c
