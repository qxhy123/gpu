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
    """stride=8 in coalescing_demo should take more cycles than stride=1."""
    ptx = (pathlib.Path(__file__).parents[2] / "examples/coalescing_demo/kernel.ptx").read_text()
    n = 1024
    a = np.arange(n, dtype=np.uint32)
    out1 = np.zeros(32, dtype=np.uint32)
    res1 = gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                     params={"A": a, "OUT": out1, "STRIDE": 1}, mode="timing")
    out8 = np.zeros(32, dtype=np.uint32)
    res8 = gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                     params={"A": a, "OUT": out8, "STRIDE": 8}, mode="timing")
    # stride=8 has 8 transactions vs 1 — at least a few extra cycles
    assert res8.metrics["cycles"] > res1.metrics["cycles"], \
        f"stride=8 cycles ({res8.metrics['cycles']}) should exceed stride=1 ({res1.metrics['cycles']})"
