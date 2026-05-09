import pandas as pd


def test_algo_efficiency_ring_vs_tree():
    from gpusim.analysis.metrics import algo_efficiency_ring_vs_tree
    df = pd.DataFrame([
        {"op_name": "allreduce", "algorithm": "ring", "n_bytes": 1024, "world_size": 4,
          "start_cycle": 0, "end_cycle": 100},
        {"op_name": "allreduce", "algorithm": "tree", "n_bytes": 32, "world_size": 4,
          "start_cycle": 0, "end_cycle": 50},
    ])
    out = algo_efficiency_ring_vs_tree(df)
    assert "ring" in out
    assert "tree" in out


def test_per_rank_communication_volume():
    from gpusim.analysis.metrics import per_rank_communication_volume
    df = pd.DataFrame([
        {"src_gpu": 0, "dst_gpu": 1, "n_bytes": 1000, "rank": 0},
        {"src_gpu": 0, "dst_gpu": 2, "n_bytes": 500, "rank": 0},
        {"src_gpu": 1, "dst_gpu": 0, "n_bytes": 800, "rank": 1},
    ])
    out = per_rank_communication_volume(df)
    assert out[0] == 1500
    assert out[1] == 800


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
