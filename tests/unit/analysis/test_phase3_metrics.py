import numpy as np, pandas as pd


def test_tc_utilization():
    from gpusim.analysis.metrics import tc_utilization
    mma = pd.DataFrame([
        {"cycle": 0, "warp_id": 0, "shape_m": 16, "shape_n": 8, "shape_k": 16},
        {"cycle": 8, "warp_id": 0, "shape_m": 16, "shape_n": 8, "shape_k": 16},
    ])
    wgmma = pd.DataFrame()
    s = tc_utilization(mma, wgmma, total_cycles=100, n_sub_cores=4)
    assert "sub_core_0" in s.columns or 0 in s.index


def test_precision_distribution():
    from gpusim.analysis.metrics import precision_distribution
    mma = pd.DataFrame([
        {"precision": "f16", "flops_count": 4096},
        {"precision": "f16", "flops_count": 4096},
        {"precision": "bf16", "flops_count": 4096},
    ])
    wgmma = pd.DataFrame()
    df = precision_distribution(mma, wgmma)
    assert df.loc["f16", "count"] == 2
    assert df.loc["f16", "flops"] == 8192


def test_effective_tflops():
    from gpusim.analysis.metrics import effective_tflops
    mma = pd.DataFrame([
        {"precision": "f16", "flops_count": 1_000_000},
    ])
    wgmma = pd.DataFrame()
    res = effective_tflops(mma, wgmma, total_cycles=1_000_000, freq_ghz=1.0)
    assert "f16" in res
    assert abs(res["f16"] - 1e-3) < 1e-9


def test_async_overlap_ratio():
    from gpusim.analysis.metrics import async_overlap_ratio
    wgmma = pd.DataFrame([
        {"kind": "ISSUE", "cycle": 0, "completion_at": 32},
    ])
    warp_state = pd.DataFrame([
        {"start": 0, "end": 16, "state": "ISSUED"},
        {"start": 17, "end": 31, "state": "WGMMA_WAIT"},
    ])
    r = async_overlap_ratio(wgmma, warp_state)
    assert 0.4 < r < 0.6


def test_mbarrier_wait_distribution():
    from gpusim.analysis.metrics import mbarrier_wait_distribution
    wgmma = pd.DataFrame([
        {"kind": "WAIT_GROUP", "cycle": 100},
        {"kind": "WAIT_GROUP", "cycle": 200},
    ])
    mbar = pd.DataFrame([
        {"kind": "FLIP", "cycle": 110},
        {"kind": "FLIP", "cycle": 215},
    ])
    s = mbarrier_wait_distribution(wgmma, mbar)
    assert isinstance(s, pd.Series)


def test_wgmma_queue_pressure():
    from gpusim.analysis.metrics import wgmma_queue_pressure
    wgmma = pd.DataFrame([
        {"kind": "ISSUE", "cycle": 0, "completion_at": 32, "warp_group_id": 0},
        {"kind": "ISSUE", "cycle": 4, "completion_at": 36, "warp_group_id": 0},
    ])
    s = wgmma_queue_pressure(wgmma, total_cycles=50)
    assert s.iloc[10] >= 1


def test_tma_bandwidth_utilization():
    from gpusim.analysis.metrics import tma_bandwidth_utilization
    tma = pd.DataFrame([
        {"cycle": 0, "completion_at": 100, "bytes_total": 1024},
    ])
    r = tma_bandwidth_utilization(tma, total_cycles=100, total_hbm_bw=10240.0)
    assert 0.05 < r < 0.15
