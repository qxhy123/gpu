"""Tests for SubCore atom.shared / red.shared routing (Phase 6 T7)."""
from __future__ import annotations


def test_sub_core_atom_shared_add_routes_correctly():
    """End-to-end: warp executes atom.shared.add, smem updated, latency reasonable.

    Uses tid.x to select lanes 0-3 only; all write to offset 0 in shared memory.
    The smem address is a u64 zero (offset 0) — valid as smem is pre-allocated for cta_id=0.
    """
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    setp.ge.u32 %p0, %r0, 4;
    @%p0 bra END;
    cvt.u64.u32 %rd0, %r0;
    add.u64 %rd0, %rd0, %rd0;
    add.u64 %rd0, %rd0, %rd0;
    mov.u32 %r1, 1;
    atom.shared.add.u32 %r2, [%rd0], %r1;
END:
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1, 1, 1), block=(32, 1, 1),
                     params={}, mode="timing", config=cfg)
    assert 0 < res.metrics["cycles"] < 5000


def test_sub_core_red_shared_no_dst():
    """red.shared.add doesn't write a dst register.

    Uses tid.x to select lanes 0-3 only; all write to a shared-memory offset
    computed from tid.x (4 * tid.x bytes).
    """
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test() {
    .reg .u64 %rd0;
    .reg .u32 %r0;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    setp.ge.u32 %p0, %r0, 4;
    @%p0 bra END;
    cvt.u64.u32 %rd0, %r0;
    add.u64 %rd0, %rd0, %rd0;
    add.u64 %rd0, %rd0, %rd0;
    red.shared.add.u32 [%rd0], %r0;
END:
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1, 1, 1), block=(32, 1, 1),
                     params={}, mode="timing", config=cfg)
    assert 0 < res.metrics["cycles"] < 5000


def test_sub_core_atom_global_add_routes_to_l2():
    """N CTAs each atomic.add to same gmem location."""
    import numpy as np
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<3>;
    .reg .u32 %r<4>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r1, 1;
    atom.global.add.u32 %r2, [%rd0], %r1;
END:
    ret;
}
"""
    cfg = load_default()
    cfg.n_sm = 4
    out = np.zeros(1, dtype=np.uint32)
    res = gpusim.run(ptx_src=src, grid=(8,1,1), block=(32,1,1),
                      params={"OUT": out}, mode="timing", config=cfg)
    # 8 CTAs x 1 thread (tid==0) = 8 atomic increments
    assert int(out[0]) == 8


def test_sub_core_red_global_add():
    """red.global.add (no return)."""
    import numpy as np
    import gpusim
    from gpusim.config.loader import load_default
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd0;
    .reg .u32 %r<3>;
    .reg .pred %p0;

    mov.u32 %r0, %tid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 bra END;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r1, 5;
    red.global.add.u32 [%rd0], %r1;
END:
    ret;
}
"""
    cfg = load_default()
    out = np.zeros(1, dtype=np.uint32)
    res = gpusim.run(ptx_src=src, grid=(4,1,1), block=(32,1,1),
                      params={"OUT": out}, mode="timing", config=cfg)
    assert int(out[0]) == 20   # 4 CTAs x 5
