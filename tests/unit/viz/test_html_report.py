from gpusim.viz.html_report import build_html
from gpusim.trace.recorder import Recorder
import pandas as pd

def test_build_html_contains_summary_and_charts(tmp_path):
    rec = Recorder()
    rec.warp_state(cycle=0, warp_id=0, state="ISSUED", pc=0)
    rec.warp_state(cycle=1, warp_id=0, state="ISSUED", pc=1)
    rec.warp_state(cycle=2, warp_id=0, state="SCOREBOARD", pc=2)
    rec.instr_issue(cycle=0, warp_id=0, pc=0, op="add.f32", src_loc=("k.ptx",1), active_mask=0xFFFFFFFF)
    rec.instr_issue(cycle=1, warp_id=0, pc=1, op="ld.global.f32", src_loc=("k.ptx",2), active_mask=0xFFFFFFFF)
    rec.smem_access(cycle=2, warp_id=0, conflict_degree=2, addresses=[0]*32)
    rec.gmem_access(cycle=1, warp_id=0, n_transactions=1, efficiency=1.0,
                    addresses=[i*4 for i in range(32)])
    rec.cta_launch(cycle=0, cta_id=0, warps=1, regs=16, smem_bytes=128)
    rec.cta_retire(cycle=10, cta_id=0)

    html = build_html(rec, kernel_name="vec_add", grid=(1,1,1), block=(32,1,1),
                      occupancy={"active_ctas":1, "bottleneck":"warps"},
                      cycles=10)
    assert "vec_add" in html
    assert "Stall breakdown" in html or "stall_breakdown" in html.lower()
    assert "<html" in html
    # plotly figures embedded as JSON inside <script>
    assert "plotly" in html.lower()
    p = tmp_path / "out.html"
    p.write_text(html)


def test_html_report_includes_phase3_sections(tmp_path):
    """When Phase 3 events present, HTML report includes §11–§14 sections."""
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.html_report import save_html
    r = Recorder()
    r.mma(cycle=0, warp_id=0, pc=0, precision="f16",
          shape_m=16, shape_n=8, shape_k=16, accum_dtype="f32",
          flops_count=4096)
    r.wgmma(kind="ISSUE", cycle=10, warp_group_id=0, pc=5,
             precision="f16", shape_m=64, shape_n=128, shape_k=16,
             completion_at=42)
    r.tma(cycle=20, completion_at=80, smem_dst=0, gmem_base=0x1000,
          dim_x=128, dim_y=64, bytes_total=16384, n_cache_lines=128,
          mbarrier_addr=0x800)
    r.mbarrier(kind="FLIP", cycle=85, cta_id=0, smem_addr=0x800,
                expected=4, arrived=4, phase=1)
    out = tmp_path / "report.html"
    save_html(r, out, kernel_name="test", grid=(1,1,1), block=(128,1,1),
              cycles=100, occupancy={"active_ctas": 1, "bottleneck": "tc"})
    html = out.read_text()
    assert "Tensor Core utilization" in html
    assert "Precision distribution" in html
    assert "wgmma" in html.lower()
    assert "mbarrier" in html.lower() or "Mbarrier" in html
