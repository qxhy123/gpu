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
