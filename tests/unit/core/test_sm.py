# tests/unit/core/test_sm.py
import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM, SMRunResult

def test_sm_runs_simple_kernel_in_timing_mode():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<3>; .reg .u64 %rd<3>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        st.global.u32 [%rd3], %r1;
        bar.sync 0;
    }
    """
    k = parse(src, "<t>")
    out = np.zeros(32, dtype=np.uint32)
    cfg = load_default()
    sm = SM(cfg=cfg)
    res = sm.run(kernel=k, grid=(1,1,1), block=(32,1,1),
                 params={"OUT": out})
    assert isinstance(res, SMRunResult)
    assert list(out) == list(range(32))
    assert res.cycles > len(k.instrs)
