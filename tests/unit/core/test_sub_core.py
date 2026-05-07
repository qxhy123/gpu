# tests/unit/core/test_sub_core.py
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sub_core import SubCore
from gpusim.core.warp import Warp, StallReason
from gpusim.core.exec import WarpFnState, GlobalMemory, SharedMemory, ParamSpace, InstrExecutor
from gpusim.core.simt_stack import SIMTStack

def _make_warp(kernel, wid=0):
    fn = WarpFnState(warp_size=32, tids=tuple(range(32)))
    return Warp(warp_id=wid, kernel=kernel, fn_state=fn,
                stack=SIMTStack(warp_size=32, entry_pc=0))

def test_subcore_issues_one_per_cycle():
    k = parse(
        ".visible .entry k() { .reg .u32 %r<4>; "
        "add.s32 %r1, %r2, %r3; add.s32 %r2, %r1, %r1; }",
        "<t>")
    cfg = load_default()
    g = GlobalMemory(); s = SharedMemory(); s.allocate_cta(0, 4096)
    p = ParamSpace({})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    sc = SubCore(sub_core_id=0, cfg=cfg, executor=ex, warps=[_make_warp(k)])
    s0 = sc.step(now=0)
    assert s0[0] is StallReason.ISSUED
    s1 = sc.step(now=1)
    assert s1[0] is StallReason.SCOREBOARD
    s2 = sc.step(now=4)
    assert s2[0] is StallReason.ISSUED

def test_subcore_idle_when_warp_done():
    k = parse(".visible .entry k() { .reg .u32 %r<2>; mov.u32 %r1, 1; }", "<t>")
    cfg = load_default()
    g = GlobalMemory(); s = SharedMemory(); s.allocate_cta(0, 4096)
    p = ParamSpace({})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    sc = SubCore(sub_core_id=0, cfg=cfg, executor=ex, warps=[_make_warp(k)])
    sc.step(now=0)
    s1 = sc.step(now=1)
    assert s1[0] is StallReason.IDLE


def test_only_one_warp_issues_per_subcore_per_cycle():
    """When multiple ready warps share a sub-core, exactly one issues per cycle;
    the others are recorded as STRUCTURAL (not ISSUED)."""
    src = ".visible .entry k() { .reg .u32 %r<4>; mov.u32 %r1, 1; mov.u32 %r2, 2; }"
    k = parse(src, "<t>")
    cfg = load_default()
    g = GlobalMemory(); s = SharedMemory(); s.allocate_cta(0, 4096)
    p = ParamSpace({})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))

    # 3 warps in one sub-core — all are ready at cycle 0, but only one can issue
    sc = SubCore(sub_core_id=0, cfg=cfg, executor=ex,
                 warps=[_make_warp(k, wid=0), _make_warp(k, wid=4), _make_warp(k, wid=8)])
    states = sc.step(now=0)
    issued_count = sum(1 for st in states if st is StallReason.ISSUED)
    structural_count = sum(1 for st in states if st is StallReason.STRUCTURAL)
    assert issued_count == 1, f"expected 1 ISSUED, got {issued_count}: {states}"
    assert structural_count == 2, f"expected 2 STRUCTURAL, got {structural_count}: {states}"
