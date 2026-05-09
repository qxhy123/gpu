import pandas as pd


def test_stream_concurrency_factor():
    from gpusim.analysis.metrics import stream_concurrency_factor
    # Stream 0: cycles 0-100; Stream 1: cycles 50-150
    df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100},
        {"stream_id": 1, "launch_cycle": 50, "complete_cycle": 150},
    ])
    factor = stream_concurrency_factor(df, total_cycles=150)
    # Average active streams per cycle: 0-50 = 1, 50-100 = 2, 100-150 = 1
    # Total active-cycles = 100 + 100 = 200; avg = 200/150 ≈ 1.333
    assert abs(factor - 1.333) < 0.01


def test_compute_memory_overlap():
    from gpusim.analysis.metrics import compute_memory_overlap
    mma_df = pd.DataFrame([
        {"cycle": 0, "stream_id": 0},
        {"cycle": 5, "stream_id": 0},
    ])
    mem_df = pd.DataFrame([
        {"cycle": 2, "stream_id": 1},
        {"cycle": 7, "stream_id": 1},
    ])
    events_dfs = {"mma": mma_df, "memory": mem_df}
    rate = compute_memory_overlap(events_dfs)
    assert 0 <= rate <= 1.0


def test_l2_bandwidth_per_stream():
    from gpusim.analysis.metrics import l2_bandwidth_per_stream
    df = pd.DataFrame([
        {"stream_id": 0}, {"stream_id": 0}, {"stream_id": 0},
        {"stream_id": 1},
    ])
    out = l2_bandwidth_per_stream(df)
    assert abs(out[0] - 0.75) < 1e-6
    assert abs(out[1] - 0.25) < 1e-6


def test_stream_fairness_jain_perfectly_fair():
    from gpusim.analysis.metrics import stream_fairness_jain
    df = pd.DataFrame([{"stream_id": 0}] * 4 + [{"stream_id": 1}] * 4)
    assert abs(stream_fairness_jain(df) - 1.0) < 1e-6


def test_stream_fairness_jain_unfair():
    from gpusim.analysis.metrics import stream_fairness_jain
    # Two streams with 8 vs 0 dispatches: but 0 means stream not present in df
    # So we need both stream_ids present. Use 7 vs 1.
    df = pd.DataFrame([{"stream_id": 0}] * 7 + [{"stream_id": 1}] * 1)
    f = stream_fairness_jain(df)
    # Jain's = (7+1)² / (2 × (49+1)) = 64/100 = 0.64
    assert abs(f - 0.64) < 0.01
