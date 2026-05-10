"""Phase 11 viz tests: §35 Graph DAG (T21) + Perfetto Graph swimlane (T22)."""
from __future__ import annotations


def test_html_report_phase11_section(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=3, n_edges=2,
                     launch_index=0, start_cycle=0, end_cycle=300)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=300, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Graph" in html or "graph" in html.lower() or "§35" in html


def test_perfetto_graph_swimlane():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.graph_launch(graph_id=0, n_nodes=3, n_edges=2,
                     launch_index=0, start_cycle=0, end_cycle=300)
    pf = build_perfetto(r)
    pids = {e.get("pid") for e in pf.get("traceEvents", [])}
    assert any("Graph" in str(p) for p in pids)
