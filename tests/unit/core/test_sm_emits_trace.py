import numpy as np
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM
from gpusim.trace.recorder import Recorder


def test_sm_run_emits_warp_state_and_issues():
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

    issues = rec.instr_issues()
    assert any(e.op.startswith("st.global") for e in issues)
    assert any(e.op == "mov.u32" for e in issues)

    gmem = rec.gmem_accesses()
    assert len(gmem) >= 1
    # the st.global.u32 should be perfectly coalesced
    assert any(g.efficiency == 1.0 for g in gmem)

    ctas = rec.cta_events()
    assert any(e.kind == "LAUNCH" for e in ctas)
    assert any(e.kind == "RETIRE" for e in ctas)
