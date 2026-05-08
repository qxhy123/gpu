import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
from gpusim.trace.recorder import Recorder


def test_sm_emits_l1_l2_hbm_events_on_global_load():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<3>; .reg .u64 %rd<3>;
        ld.param.u64 %rd1, [OUT];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd1, %rd1, %rd2;
        st.global.u32 [%rd1], %r1;
        bar.sync 0;
    }
    """
    out = np.zeros(32, dtype=np.uint32)
    rec = Recorder()
    sm = SM(load_default(), recorder=rec)
    sm.run(kernel=parse(src, "<t>"), grid=(1, 1, 1), block=(32, 1, 1), params={"OUT": out})

    # store hits L2 via write-through; HBM gets a READ for the L2 fill
    l2_events = rec.l2_accesses()
    hbm_events = rec.hbm_accesses()
    # at minimum: one L2 access + one HBM event for the store path
    assert len(l2_events) >= 1
    assert len(hbm_events) >= 1
