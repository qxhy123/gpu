"""Phase 10 HTML §33/§34 + Perfetto NVLink/Collective swimlane tests."""
from __future__ import annotations


def test_html_report_phase10_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.collective(op_name="allreduce", algorithm="ring", n_bytes=1024,
                   world_size=4, start_cycle=0, end_cycle=100, n_steps=6)
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=256,
                        start_cycle=0, end_cycle=10)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(32,1,1),
              cycles=200, occupancy={"active_ctas": 1, "bottleneck": "none"})
    html = out.read_text()
    assert "Collective" in html or "collective" in html.lower() \
            or "NVLink" in html or "nvlink" in html.lower()


def test_perfetto_nvlink_swimlane():
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import build_perfetto
    r = Recorder()
    r.nvlink_transfer(src_gpu=0, dst_gpu=1, n_bytes=1024,
                        start_cycle=0, end_cycle=100, rank=0, op_name="allreduce")
    r.collective(op_name="allreduce", algorithm="ring", n_bytes=1024,
                   world_size=4, start_cycle=0, end_cycle=100, n_steps=6)
    pf = build_perfetto(r)
    pids = {e.get("pid") for e in pf.get("traceEvents", [])}
    assert any("NVLink" in str(p) for p in pids)
