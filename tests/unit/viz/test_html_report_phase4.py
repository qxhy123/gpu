def test_html_report_phase4_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.cta_dispatch(cycle=0, cta_id=0, sm_id=0)
    r.l2_mshr(kind="ALLOC", cycle=1, line_addr=0, sm_id=0)
    r.bulk_store(kind="ISSUE", cycle=2, warp_group_id=0, sm_id=0,
                   completion_at=20, bytes_total=128, pc=5,
                   smem_src=0, gmem_base=0)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(1,1,1), block=(128,1,1),
              cycles=100, occupancy={"active_ctas":1,"bottleneck":"tc"})
    html = out.read_text()
    assert "Per-SM" in html or "per-sm" in html.lower()
    assert "CTA" in html
    assert "MSHR" in html.lower() or "mshr" in html.lower()
    assert "BulkStore" in html or "bulk" in html.lower()
