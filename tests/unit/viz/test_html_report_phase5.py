def test_html_report_phase5_sections(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.cluster_dispatch(cycle=0, cluster_id=0, cluster_size=2,
                         sm_ids=(0,1), cta_ids=(0,1), queue_position=0)
    r.cluster_barrier(kind="ARRIVE", cycle=10, cluster_id=0,
                        cta_id=0, rank=0, sm_id=0, arrived_count=1)
    r.cluster_barrier(kind="WAIT_RELEASE", cycle=20, cluster_id=0,
                        cta_id=0, rank=0, sm_id=0)
    out = tmp_path / "rpt.html"
    save_html(r, out, kernel_name="t", grid=(2,1,1), block=(32,1,1),
              cycles=100, occupancy={"active_ctas":1, "bottleneck":"none"})
    html = out.read_text()
    assert "Cluster" in html
    assert "barrier" in html.lower() or "Barrier" in html
