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
