def test_html_report_phase9_combined_overlap_section(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k0", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1)
    r.kernel_launch(stream_id=1, kernel_name="k1", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=50, complete_cycle=150, n_ctas=1)
    r.stream_event(cycle=100, event_id=1, stream_id=0, op="record")
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Combined" in html or "combined" in html.lower() or "§32" in html


def test_perfetto_record_wait_async_arrows():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.stream_event(cycle=50, event_id=1, stream_id=0, op="record")
    r.stream_event(cycle=100, event_id=1, stream_id=1, op="wait_satisfied")
    pf = build_perfetto(r)
    cats = [e.get("cat") for e in pf.get("traceEvents", [])]
    # Should emit a "stream_event_arrow" category for the record→wait pair
    assert any("stream_event_arrow" in str(c) for c in cats)
