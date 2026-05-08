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


def test_subcore_issues_global_load_through_l1():
    """ld.global goes through L1 cache; first access misses, returns appropriate ready_at."""
    from gpusim.core.cache.l1 import L1Cache
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.core.hbm import HBM
    import numpy as np

    src = """
    .visible .entry k(.param .u64 A) {
        .reg .u32 %r<3>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        ld.param.u64 %rd1, [A];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 2;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd1, %rd1, %rd2;
        ld.global.f32 %f1, [%rd1];
    }
    """
    k = parse(src, "<t>")
    cfg = load_default()
    arr = np.arange(32, dtype=np.float32)
    g = GlobalMemory()
    g.bind("A", arr)
    s = SharedMemory()
    s.allocate_cta(0, 4096)
    p = ParamSpace({"A": g.address_of("A")})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    fn = WarpFnState(warp_size=32, tids=tuple(range(32)))
    w = Warp(warp_id=0, kernel=k, fn_state=fn,
             stack=SIMTStack(warp_size=32, entry_pc=0))

    hbm = HBM(cfg.hbm)
    l2 = L2Cache(cfg.cache, hbm)
    l1 = L1Cache(cfg.cache, l2)

    sc = SubCore(sub_core_id=0, cfg=cfg, executor=ex, warps=[w], l1=l1)
    for cycle in range(2000):
        sc.step(now=cycle)
        l1.install_completed_lines(now=cycle)
        if w.finished or (w.stack and w.stack.is_done()):
            break
    assert w.finished or w.stack.is_done()


def test_subcore_emits_mshr_full_when_pool_saturated():
    from gpusim.config.schema import CacheConfig
    from gpusim.core.cache.l1 import L1Cache
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.core.hbm import HBM
    import numpy as np

    cfg = load_default()
    cfg.cache = CacheConfig(mshr_slots=1)  # tiny pool — fills after 1 miss
    # Use stride-128 (shl by 7 = tid*128) so all 32 threads touch different
    # 128-byte cache lines. The first ld.global allocates the 1 MSHR slot; the
    # second distinct line in the same warp access causes Reject → MSHR_FULL.
    # Two sequential ld.global instructions ensure the second one sees a full pool.
    src = """
    .visible .entry k(.param .u64 A) {
        .reg .u32 %r<5>; .reg .u64 %rd<6>; .reg .f32 %f<5>;
        ld.param.u64 %rd1, [A];
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 7;
        cvt.u64.u32 %rd2, %r2;
        add.u64 %rd3, %rd1, %rd2;
        ld.global.f32 %f1, [%rd3];
        ld.global.f32 %f2, [%rd3];
        ld.global.f32 %f3, [%rd3];
    }
    """
    k = parse(src, "<t>")
    # stride=128 bytes per thread * 32 threads = 4096 bytes needed
    arr = np.arange(1024, dtype=np.float32)
    g = GlobalMemory(); g.bind("A", arr)
    s = SharedMemory(); s.allocate_cta(0, 4096)
    p = ParamSpace({"A": g.address_of("A")})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                       ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    fn = WarpFnState(warp_size=32, tids=tuple(range(32)))
    w = Warp(warp_id=0, kernel=k, fn_state=fn,
             stack=SIMTStack(warp_size=32, entry_pc=0))
    hbm = HBM(cfg.hbm)
    l2 = L2Cache(cfg.cache, hbm)
    l1 = L1Cache(cfg.cache, l2)

    sc = SubCore(sub_core_id=0, cfg=cfg, executor=ex, warps=[w], l1=l1)
    saw_mshr_full = False
    for cycle in range(2000):
        states = sc.step(now=cycle)
        l1.install_completed_lines(now=cycle)
        if states[0] is StallReason.MSHR_FULL:
            saw_mshr_full = True
        if w.finished or (w.stack and w.stack.is_done()):
            break
    assert saw_mshr_full, "expected at least one MSHR_FULL stall"


def test_subcore_issues_sync_mma_with_tc_latency():
    """sync mma reserves TC FU and marks dst regs ready at now + tc_mma_latency."""
    import numpy as np
    from gpusim.config.schema import SMConfig
    from gpusim.core.warp import Warp
    from gpusim.core.sub_core import SubCore
    from gpusim.core.exec import (
        WarpFnState, GlobalMemory, SharedMemory, ParamSpace, InstrExecutor,
    )
    from gpusim.core.simt_stack import SIMTStack
    from gpusim.frontend.parser import parse

    src = """
.entry test()
{
    .reg .f32 %d<4>;
    .reg .f16 %a<8>;
    .reg .f16 %b<4>;
    .reg .f32 %c<4>;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {%d0, %d1, %d2, %d3},
        {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7},
        {%b0, %b1, %b2, %b3},
        {%c0, %c1, %c2, %c3};
}
"""
    k = parse(src, "<test>")
    cfg = SMConfig()
    g = GlobalMemory(); s = SharedMemory()
    p = ParamSpace({})
    ex = InstrExecutor(kernel=k, gmem=g, smem=s, params=p, cta_id=0,
                        ctaid=(0,0,0), nctaid=(1,1,1), ntid=(32,1,1))
    fn = WarpFnState(warp_size=32, tids=tuple(range(32)))
    w = Warp(warp_id=0, kernel=k, fn_state=fn,
              stack=SIMTStack(warp_size=32, entry_pc=0), cta_id=0)
    sc = SubCore(0, cfg, ex, [w])
    sc.step(now=0)
    # After issuing mma at cycle 0, dst reg %d0 should be ready at cycle 8 (tc_mma_latency)
    assert w.scoreboard.has_pending("d0", now=4) is True
    assert w.scoreboard.has_pending("d0", now=8) is False
