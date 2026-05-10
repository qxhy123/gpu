import pandas as pd


def test_reduce_scatter_step_count():
    from gpusim.analysis.metrics import reduce_scatter_step_count
    df = pd.DataFrame([
        {"op_name": "reduce_scatter", "algorithm": "ring", "n_steps": 3,
          "n_bytes": 256, "world_size": 4, "start_cycle": 0, "end_cycle": 100},
        {"op_name": "reduce_scatter", "algorithm": "ring", "n_steps": 7,
          "n_bytes": 512, "world_size": 8, "start_cycle": 100, "end_cycle": 200},
    ])
    out = reduce_scatter_step_count(df)
    assert 3 in out
    assert 7 in out


def test_dist_api_call_breakdown():
    from gpusim.analysis.metrics import dist_api_call_breakdown
    df = pd.DataFrame([
        {"op_name": "allreduce", "algorithm": "ring"},
        {"op_name": "allreduce", "algorithm": "tree"},
        {"op_name": "broadcast", "algorithm": "linear"},
        {"op_name": "reduce_scatter", "algorithm": "ring"},
    ])
    out = dist_api_call_breakdown(df)
    assert out["allreduce"] == 2
    assert out["broadcast"] == 1
    assert out["reduce_scatter"] == 1
