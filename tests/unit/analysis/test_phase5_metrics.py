import pandas as pd


def test_cluster_dispatch_latency():
    from gpusim.analysis.metrics import cluster_dispatch_latency
    df = pd.DataFrame([
        {"cycle": 0, "cluster_id": 0},
        {"cycle": 5, "cluster_id": 1},
    ])
    s = cluster_dispatch_latency(df, cta_launch_df=None)
    assert isinstance(s, pd.Series)


def test_cluster_barrier_wait_distribution():
    from gpusim.analysis.metrics import cluster_barrier_wait_distribution
    df = pd.DataFrame([
        {"kind": "ARRIVE", "cycle": 10, "cluster_id": 0},
        {"kind": "ARRIVE", "cycle": 15, "cluster_id": 0},
        {"kind": "WAIT_RELEASE", "cycle": 20, "cluster_id": 0},
    ])
    s = cluster_barrier_wait_distribution(df)
    assert isinstance(s, pd.Series)


def test_dsmem_remote_access_rate():
    from gpusim.analysis.metrics import dsmem_remote_access_rate
    instr_df = pd.DataFrame([
        {"op": "ld.shared.f32"},
        {"op": "ld.shared::cluster.f32"},
        {"op": "st.shared::cluster.f32"},
        {"op": "st.shared.f32"},
    ])
    rate = dsmem_remote_access_rate(instr_df)
    assert abs(rate - 0.5) < 1e-6
