import numpy as np
import pathlib
import gpusim
from gpusim.frontend.parser import parse
from gpusim.config.loader import load_default
from gpusim.core.sm import SM


def _run(src, params, grid=(1,1,1), block=(32,1,1), cfg=None):
    k = parse(src, "<t>")
    sm = SM(cfg or load_default())
    return sm.run(kernel=k, grid=grid, block=block, params=params)


def test_stride_32_word_shared_is_32way_conflict():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<5>; .reg .u64 %rd<3>; .reg .f32 %f<2>;
        mov.u32 %r1, %tid.x;
        shl.b32 %r2, %r1, 7;
        cvt.u64.u32 %rd1, %r2;
        cvt.f32.s32 %f1, %r1;
        st.shared.f32 [%rd1], %f1;
        bar.sync 0;
    }
    """
    out = np.zeros(32, dtype=np.float32)
    res_conflict = _run(src, {"OUT": out})
    src_no = src.replace("shl.b32 %r2, %r1, 7;", "shl.b32 %r2, %r1, 2;")
    out2 = np.zeros(32, dtype=np.float32)
    res_no = _run(src_no, {"OUT": out2})
    assert res_conflict.cycles >= res_no.cycles + 25


def test_stride_2_global_efficiency_50pct():
    from gpusim.core.gmem import coalescing_info
    addrs = [i * 8 for i in range(32)]
    info = coalescing_info(addrs)
    assert info.n_transactions == 2
    assert abs(info.efficiency - 0.5) < 1e-9


def test_one_warp_kernel_ipc_le_1():
    src = """
    .visible .entry k(.param .u64 OUT) {
        .reg .u32 %r<5>;
        mov.u32 %r1, 1;
        add.s32 %r2, %r1, %r1;
        add.s32 %r3, %r2, %r2;
        bar.sync 0;
    }
    """
    out = np.zeros(1, dtype=np.uint32)
    res = _run(src, {"OUT": out}, block=(32,1,1))
    assert res.cycles >= 4


def test_strided_global_costs_more_cycles_than_coalesced():
    """stride=8 should take more cycles than stride=1 when MSHR slots are limited.

    With a full L1 cache (16 MSHR slots), both strides complete in the same
    number of cycles because all cache-line misses are issued in parallel.
    With mshr_slots=4 and stride=8 needing 8 distinct cache lines, MSHR_FULL
    stalls force the warp to retry, causing more cycles than stride=1 (1 line).
    """
    from gpusim.config.schema import CacheConfig
    ptx = (pathlib.Path(__file__).parents[2] / "examples/coalescing_demo/kernel.ptx").read_text()
    k = parse(ptx, "coal_demo")
    n = 1024
    a = np.arange(n, dtype=np.uint32)

    cfg = load_default()
    cfg.cache = CacheConfig(mshr_slots=4)  # 4 slots: stride=1 (1 line) fits; stride=8 (8 lines) stalls

    sm1 = SM(cfg)
    out1 = np.zeros(32, dtype=np.uint32)
    res1 = sm1.run(kernel=k, grid=(1,1,1), block=(32,1,1),
                   params={"A": a, "OUT": out1, "STRIDE": 1})

    sm8 = SM(cfg)
    out8 = np.zeros(32, dtype=np.uint32)
    res8 = sm8.run(kernel=k, grid=(1,1,1), block=(32,1,1),
                   params={"A": a, "OUT": out8, "STRIDE": 8})

    # stride=8 causes MSHR_FULL stalls (8 lines needed > 4 slots) → more cycles
    assert res8.cycles > res1.cycles, \
        f"stride=8 cycles ({res8.cycles}) should exceed stride=1 ({res1.cycles})"
