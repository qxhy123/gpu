"""T16: Phase 11 analysis metrics — graph_replay_amortization, graph_dag_depth,
graph_node_type_breakdown."""
from __future__ import annotations
import pandas as pd


def test_graph_replay_amortization():
    from gpusim.analysis.metrics import graph_replay_amortization
    df = pd.DataFrame([
        {"launch_index": 0, "start_cycle": 0, "end_cycle": 300},
        {"launch_index": 1, "start_cycle": 300, "end_cycle": 590},
        {"launch_index": 2, "start_cycle": 590, "end_cycle": 870},
    ])
    out = graph_replay_amortization(df, single_kernel_baseline_cycles=150)
    assert "avg_cycles_per_replay" in out
    assert "amortization_factor" in out


def test_graph_dag_depth_linear():
    from gpusim.analysis.metrics import graph_dag_depth
    from gpusim.graph.graph import Graph
    g = Graph()
    a = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="ka")
    b = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="kb")
    c = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="kc")
    g.add_dependency(a, b)
    g.add_dependency(b, c)
    assert graph_dag_depth(g) == 3


def test_graph_node_type_breakdown():
    from gpusim.analysis.metrics import graph_node_type_breakdown
    from gpusim.graph.graph import Graph
    import numpy as np
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    g.add_kernel_node(ptx_src="y", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k2")
    g.add_memcpy_node(src=np.zeros(8), dst=np.zeros(8), n_bytes=32)
    out = graph_node_type_breakdown(g)
    assert out["kernel"] == 2
    assert out["memcpy"] == 1
