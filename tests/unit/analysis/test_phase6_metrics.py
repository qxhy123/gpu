import pandas as pd


def test_atomic_throughput_per_line():
    from gpusim.analysis.metrics import atomic_throughput_per_line
    df = pd.DataFrame([
        {"cycle": 0, "line_addr": 0x1000},
        {"cycle": 10, "line_addr": 0x1000},
        {"cycle": 5, "line_addr": 0x2000},
    ])
    out = atomic_throughput_per_line(df, total_cycles=100)
    assert isinstance(out, pd.DataFrame)
    assert (out["line_addr"] == 0x1000).any()


def test_atomic_serialization_overhead():
    from gpusim.analysis.metrics import atomic_serialization_overhead
    df = pd.DataFrame([
        {"cycle": 0, "latency": 50},
        {"cycle": 0, "latency": 50},
    ])
    rate = atomic_serialization_overhead(df, total_cycles=100)
    # 2 atomics × 50 latency = 100; total 100 → rate ~1.0
    assert 0 <= rate <= 1.0


def test_atom_vs_red_ratio():
    from gpusim.analysis.metrics import atom_vs_red_ratio
    df = pd.DataFrame([
        {"kind": "ATOM"}, {"kind": "ATOM"}, {"kind": "ATOM"},
        {"kind": "RED"},
    ])
    r = atom_vs_red_ratio(df)
    assert abs(r["atom"] - 0.75) < 1e-6
    assert abs(r["red"] - 0.25) < 1e-6


def test_cooperative_epilogue_overlap():
    from gpusim.analysis.metrics import cooperative_epilogue_overlap
    bulk_df = pd.DataFrame([
        {"kind": "ISSUE", "cycle": 0, "completion_at": 50},
    ])
    mma_df = pd.DataFrame([
        {"cycle": 10},
        {"cycle": 20},
    ])
    r = cooperative_epilogue_overlap(bulk_df, mma_df)
    assert 0 <= r <= 1.0
