def test_html_report_phase7_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k0", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1)
    r.kernel_launch(stream_id=1, kernel_name="k1", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=50, complete_cycle=150, n_ctas=1)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Stream" in html or "stream" in html.lower()


def test_perfetto_kernel_launch_stream_swimlane():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.kernel_launch(stream_id=0, kernel_name="k0", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=0, complete_cycle=100, n_ctas=1)
    r.kernel_launch(stream_id=1, kernel_name="k1", grid=(1,1,1), block=(32,1,1),
                     launch_cycle=50, complete_cycle=150, n_ctas=1)
    pf = build_perfetto(r)
    # Check that some events use pid="Stream-0" and pid="Stream-1"
    pids = {e.get("pid") for e in pf.get("traceEvents", [])}
    assert any("Stream-0" in str(p) for p in pids)
    assert any("Stream-1" in str(p) for p in pids)
