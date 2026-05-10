"""T14: GraphLaunch trace event, recorder, and parquet writer tests."""
from __future__ import annotations


def test_recorder_records_graph_launch():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=3, n_edges=2,
                     launch_index=0, start_cycle=0, end_cycle=300)
    assert len(r.graph_launch_events) == 1
    e = r.graph_launch_events[0]
    assert e.n_nodes == 3
    assert e.launch_index == 0


def test_recorder_writes_graph_launch_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=2, n_edges=1,
                     launch_index=0, start_cycle=0, end_cycle=200)
    write_parquet(r, tmp_path)
    assert (tmp_path / "graph_launch.parquet").exists()
