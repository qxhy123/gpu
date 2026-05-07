import numpy as np
from gpusim.frontend.parser import parse
from gpusim.frontend.ir import PtxType
from gpusim.core.exec import (
    WarpFnState, GlobalMemory, SharedMemory, ParamSpace, InstrExecutor,
)

KT = """
.visible .entry k(.param .u64 A, .param .u32 N) {{
    .reg .u32 %r<8>;
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .pred %p<2>;
    {body}
}}
"""

def make_ctx(body, params=None, gmem_arrays=None):
    k = parse(KT.format(body=body), "<t>")
    g = GlobalMemory()
    if gmem_arrays:
        for name, arr in gmem_arrays.items():
            params = dict(params or {})
            params[name] = g.bind(name, arr)
    s = SharedMemory()
    s.allocate_cta(0, 4096)
    p = ParamSpace(params or {})
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    return k, w, ex

def test_add_s32_per_lane():
    k, w, ex = make_ctx("mov.u32 %r1, %tid.x; add.s32 %r2, %r1, 100; bra END; END: bar.sync 0;")
    while w.pc < len(k.instrs):
        ex.execute(w, k.instrs[w.pc])
        w.pc += 1
    for lane in range(32):
        assert w.threads[lane].get_s32("r2") == lane + 100

def test_setp_lt_sets_predicate_per_lane():
    body = "mov.u32 %r1, %tid.x; setp.lt.s32 %p1, %r1, 8;"
    k, w, ex = make_ctx(body)
    while w.pc < len(k.instrs):
        ex.execute(w, k.instrs[w.pc])
        w.pc += 1
    for lane in range(32):
        assert w.threads[lane].get_pred("p1") is (lane < 8)

def test_ld_global_f32():
    arr = np.arange(32, dtype=np.float32)
    body = """
        ld.param.u64 %rd1, [A];
        mov.u32 %r1, %tid.x;
        mul.lo.s32 %r2, %r1, 4;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        ld.global.f32 %f1, [%rd3];
    """
    k, w, ex = make_ctx(body, gmem_arrays={"A": arr})
    while w.pc < len(k.instrs):
        ex.execute(w, k.instrs[w.pc])
        w.pc += 1
    for lane in range(32):
        assert w.threads[lane].get_f32("f1") == float(lane)
