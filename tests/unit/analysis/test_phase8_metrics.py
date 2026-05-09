import pandas as pd


def test_cross_stream_concurrency_gain():
    from gpusim.analysis.metrics import cross_stream_concurrency_gain
    df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100},
        {"stream_id": 1, "launch_cycle": 0, "complete_cycle": 100},
    ])
    # 2 launches, each 100 cycles, total = 100 cycles → gain = 200/100 = 2.0
    gain = cross_stream_concurrency_gain(df, total_cycles=100)
    assert abs(gain - 2.0) < 0.01


def test_cross_stream_concurrency_gain_no_overlap():
    from gpusim.analysis.metrics import cross_stream_concurrency_gain
    df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100},
        {"stream_id": 1, "launch_cycle": 100, "complete_cycle": 200},
    ])
    # 200 cycles total wall, 200 cycles total work → gain = 1.0
    gain = cross_stream_concurrency_gain(df, total_cycles=200)
    assert abs(gain - 1.0) < 0.01


def test_priority_dispatch_share():
    from gpusim.analysis.metrics import priority_dispatch_share
    df = pd.DataFrame([
        {"stream_id": 0}, {"stream_id": 0}, {"stream_id": 0}, {"stream_id": 0},
        {"stream_id": 1}, {"stream_id": 1},
        {"stream_id": 2},
    ])
    # 4 high (sid=0), 2 normal (sid=1), 1 low (sid=2) → 4/7, 2/7, 1/7
    stream_priority = {0: "high", 1: "normal", 2: "low"}
    out = priority_dispatch_share(df, stream_priority)
    assert abs(out["high"] - 4/7) < 0.01
    assert abs(out["normal"] - 2/7) < 0.01
    assert abs(out["low"] - 1/7) < 0.01


def test_event_wait_cycles_per_stream():
    from gpusim.analysis.metrics import event_wait_cycles_per_stream
    df = pd.DataFrame([
        {"cycle": 10, "event_id": 1, "stream_id": 1, "op": "wait_start"},
        {"cycle": 60, "event_id": 1, "stream_id": 1, "op": "wait_satisfied"},
    ])
    out = event_wait_cycles_per_stream(df)
    assert out[1] == 50


def test_event_chain_critical_path():
    from gpusim.analysis.metrics import event_chain_critical_path
    se_df = pd.DataFrame([
        {"cycle": 100, "event_id": 1, "stream_id": 0, "op": "record"},
        {"cycle": 100, "event_id": 1, "stream_id": 1, "op": "wait_satisfied"},
    ])
    kl_df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100, "kernel_name": "a"},
        {"stream_id": 1, "launch_cycle": 100, "complete_cycle": 200, "kernel_name": "b"},
    ])
    cp = event_chain_critical_path(se_df, kl_df)
    # a (100) → ev1 → b (100) = 200 total
    assert cp == 200


def test_l2_window_hit_rate_per_stream():
    from gpusim.analysis.metrics import l2_window_hit_rate_per_stream
    df = pd.DataFrame([
        {"stream_id": 0, "hit": True}, {"stream_id": 0, "hit": True},
        {"stream_id": 0, "hit": False}, {"stream_id": 1, "hit": False},
    ])
    out = l2_window_hit_rate_per_stream(df)
    assert abs(out[0] - 2/3) < 0.01
    assert abs(out[1] - 0.0) < 0.01


def test_l2_window_protection_efficiency():
    from gpusim.analysis.metrics import l2_window_protection_efficiency
    df = pd.DataFrame([
        {"hit": True, "in_window": True}, {"hit": True, "in_window": True},
        {"hit": True, "in_window": False}, {"hit": False, "in_window": False},
    ])
    eff = l2_window_protection_efficiency(df)
    # 2 in-window hits out of 3 total hits = 0.67
    assert abs(eff - 2/3) < 0.01
