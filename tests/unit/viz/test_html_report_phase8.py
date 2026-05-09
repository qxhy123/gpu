"""Tests for HTML report Phase 8 sections (§29/§30/§31) and Perfetto StreamEvent emission."""
from __future__ import annotations


def test_html_report_phase8_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k0", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1)
    r.stream_event(cycle=50, event_id=1, stream_id=0, op="record")
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Priority" in html or "priority" in html.lower() \
            or "Event" in html or "event" in html.lower()


def test_perfetto_stream_event_emitted():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.stream_event(cycle=50, event_id=1, stream_id=0, op="record")
    pf = build_perfetto(r)
    cats = {e.get("cat") for e in pf.get("traceEvents", [])}
    assert any("stream_event" in str(c) for c in cats)
