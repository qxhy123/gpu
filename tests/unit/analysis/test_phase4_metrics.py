import numpy as np, pandas as pd


def test_per_sm_utilization_returns_dataframe_per_sm():
    from gpusim.analysis.metrics import per_sm_utilization
    warp_state = pd.DataFrame([
        {"sm_id": 0, "start": 0, "end": 50, "state": "ISSUED"},
        {"sm_id": 1, "start": 0, "end": 25, "state": "ISSUED"},
    ])
    df = per_sm_utilization(warp_state, total_cycles=100, n_sm=2)
    assert df.shape[0] >= 1


def test_cta_to_sm_mapping():
    from gpusim.analysis.metrics import cta_to_sm_mapping
    dispatch_df = pd.DataFrame([
        {"cycle": 0, "cta_id": 0, "sm_id": 0},
        {"cycle": 0, "cta_id": 1, "sm_id": 1},
        {"cycle": 5, "cta_id": 2, "sm_id": 0},
    ])
    mapping = cta_to_sm_mapping(dispatch_df)
    assert len(mapping) == 3
    assert "sm_id" in mapping.columns


def test_cta_dispatch_latency():
    from gpusim.analysis.metrics import cta_dispatch_latency
    dispatch_df = pd.DataFrame([
        {"cycle": 0, "cta_id": 0, "sm_id": 0},
        {"cycle": 10, "cta_id": 1, "sm_id": 1},
    ])
    s = cta_dispatch_latency(dispatch_df, cta_launch_df=None)
    assert isinstance(s, pd.Series)


def test_l2_cross_sm_hit_rate():
    from gpusim.analysis.metrics import l2_cross_sm_hit_rate
    l2_events = pd.DataFrame([
        {"kind": "HIT", "origin_sm": 0, "hit_sm": 0},
        {"kind": "HIT", "origin_sm": 0, "hit_sm": 1},
        {"kind": "HIT", "origin_sm": 0, "hit_sm": 2},
        {"kind": "MISS_LOAD", "origin_sm": 3, "hit_sm": 3},
    ])
    rate = l2_cross_sm_hit_rate(l2_events)
    assert abs(rate - 2/3) < 1e-6


def test_l2_mshr_pressure():
    from gpusim.analysis.metrics import l2_mshr_pressure
    df = pd.DataFrame([
        {"kind": "ALLOC", "cycle": 0, "line_addr": 1},
        {"kind": "ALLOC", "cycle": 5, "line_addr": 2},
        {"kind": "RELEASE", "cycle": 10, "line_addr": 1},
    ])
    s = l2_mshr_pressure(df, total_cycles=20)
    assert isinstance(s, pd.Series)
    assert s.iloc[7] == 2


def test_bulk_store_async_overlap_ratio():
    from gpusim.analysis.metrics import bulk_store_async_overlap_ratio
    bulk_df = pd.DataFrame([
        {"kind": "ISSUE", "cycle": 0, "completion_at": 20},
    ])
    warp_state = pd.DataFrame([
        {"start": 0, "end": 10, "state": "ISSUED"},
        {"start": 11, "end": 20, "state": "BULK_STORE_WAIT"},
    ])
    r = bulk_store_async_overlap_ratio(bulk_df, warp_state)
    assert 0.4 < r < 0.6
