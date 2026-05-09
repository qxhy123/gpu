import pandas as pd


def test_actual_cross_grid_overlap_cycles():
    from gpusim.analysis.metrics import actual_cross_grid_overlap_cycles
    df = pd.DataFrame([
        {"stream_id": 0, "launch_cycle": 0, "complete_cycle": 100},
        {"stream_id": 1, "launch_cycle": 50, "complete_cycle": 150},
    ])
    # Cycles 50-100 have both active → ~50 overlap cycles
    overlap = actual_cross_grid_overlap_cycles(df, total_cycles=150)
    assert 49 <= overlap <= 51   # allow boundary inclusivity


def test_l2_eviction_protected_count_empty():
    from gpusim.analysis.metrics import l2_eviction_protected_count
    out = l2_eviction_protected_count(None)
    assert out == {}
