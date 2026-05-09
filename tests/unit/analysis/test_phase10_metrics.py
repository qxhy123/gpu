import pandas as pd


def test_nvlink_bandwidth_utilization():
    from gpusim.analysis.metrics import nvlink_bandwidth_utilization
    df = pd.DataFrame([
        {"src_gpu": 0, "dst_gpu": 1, "n_bytes": 1000, "start_cycle": 0, "end_cycle": 100},
        {"src_gpu": 1, "dst_gpu": 0, "n_bytes": 500, "start_cycle": 0, "end_cycle": 50},
    ])
    out = nvlink_bandwidth_utilization(df, total_cycles=100)
    assert (0, 1) in out
    assert out[(0, 1)] == 10.0


def test_collective_op_breakdown():
    from gpusim.analysis.metrics import collective_op_breakdown
    df = pd.DataFrame([
        {"op_name": "allreduce", "algorithm": "ring", "n_bytes": 1024,
          "world_size": 4, "start_cycle": 0, "end_cycle": 100, "n_steps": 6},
        {"op_name": "broadcast", "algorithm": "linear", "n_bytes": 64,
          "world_size": 4, "start_cycle": 100, "end_cycle": 150, "n_steps": 3},
    ])
    out = collective_op_breakdown(df)
    assert ("allreduce", "ring") in out
    assert out[("allreduce", "ring")] == 100
