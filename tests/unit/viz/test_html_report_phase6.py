"""Tests for T21: HTML §21/§22 + Perfetto Atomic swimlane."""
from __future__ import annotations


def test_html_report_phase6_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.atomic(cycle=0, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000, latency=50)
    r.atomic(cycle=10, sm_id=1, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000, latency=50)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas":1, "bottleneck":"none"})
    html = out.read_text()
    assert "Atomic" in html or "atomic" in html.lower()


def test_html_report_phase6_section21_heading(tmp_path):
    """§21 heading appears when atomic events are present."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.atomic(cycle=5, sm_id=0, warp_id=1, kind="ATOM",
              op="min", space="global", line_addr=0x2000, latency=30)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="k", grid=(1,1,1), block=(32,1,1),
              cycles=100, occupancy={"active_ctas":1, "bottleneck":"none"})
    html = out.read_text()
    assert "§21" in html
    assert "Atomic" in html


def test_html_report_phase6_no_atomic_no_section(tmp_path):
    """§21 block is absent when there are no atomic events."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="k", grid=(1,1,1), block=(32,1,1),
              cycles=100, occupancy={"active_ctas":1, "bottleneck":"none"})
    html = out.read_text()
    assert "§21" not in html


def test_html_report_phase6_section22_heading(tmp_path):
    """§22 heading appears when bulk_store events are present."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.bulk_store(kind="ISSUE", cycle=0, warp_group_id=0, sm_id=0,
                 bytes_total=512, completion_at=20)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="k", grid=(1,1,1), block=(32,1,1),
              cycles=100, occupancy={"active_ctas":1, "bottleneck":"none"})
    html = out.read_text()
    assert "§22" in html


def test_html_report_phase6_hot_lines_table(tmp_path):
    """Hot lines table appears when multiple atomics share a line_addr."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    for i in range(5):
        r.atomic(cycle=i*10, sm_id=i % 2, warp_id=0, kind="ATOM",
                 op="add", space="global", line_addr=0x1000, latency=20)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="k", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas":1, "bottleneck":"none"})
    html = out.read_text()
    assert "Hot lines" in html or "hot" in html.lower()


def test_perfetto_atomic_track():
    """build_perfetto emits 'atomic' category events for AtomicEvents."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.atomic(cycle=5, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000, latency=50)
    r.atomic(cycle=20, sm_id=1, warp_id=1, kind="RED",
              op="min", space="global", line_addr=0x2000, latency=30,
              n_lanes=16, queue_depth_before=2)
    trace = build_perfetto(r)
    atomic_events = [e for e in trace["traceEvents"] if e.get("cat") == "atomic"]
    assert len(atomic_events) == 2
    ev = atomic_events[0]
    assert ev["ph"] == "X"
    assert ev["pid"] == "Atomic"
    assert ev["dur"] >= 1
    assert "line_addr" in ev["args"]
    assert "sm_id" in ev["args"]
    assert "n_lanes" in ev["args"]
    assert "queue_depth_before" in ev["args"]


def test_perfetto_atomic_event_name():
    """Atomic event names follow kind.space.op pattern."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.atomic(cycle=0, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000, latency=10)
    trace = build_perfetto(r)
    atomic_events = [e for e in trace["traceEvents"] if e.get("cat") == "atomic"]
    assert atomic_events[0]["name"] == "atom.global.add"


def test_perfetto_atomic_duration_floor():
    """Atomic events with latency=0 still get dur>=1."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.atomic(cycle=0, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000, latency=0)
    trace = build_perfetto(r)
    atomic_events = [e for e in trace["traceEvents"] if e.get("cat") == "atomic"]
    assert atomic_events[0]["dur"] >= 1
