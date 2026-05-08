import json
from gpusim.viz.perfetto import build_perfetto
from gpusim.trace.recorder import Recorder

def test_perfetto_emits_warp_tracks_and_slices():
    rec = Recorder()
    rec.warp_state(cycle=0, warp_id=0, state="ISSUED", pc=0)
    rec.warp_state(cycle=1, warp_id=0, state="SCOREBOARD", pc=1)
    rec.instr_issue(cycle=0, warp_id=0, pc=0, op="add.f32",
                    src_loc=("k.ptx",1), active_mask=0xFFFFFFFF)
    rec.div_push(cycle=2, warp_id=0, pc=2, rpc=10, taken_mask=0xFF)
    rec.bar_reach(cycle=5, cta_id=0)
    rec.bar_release(cycle=10, cta_id=0)

    obj = build_perfetto(rec)
    assert obj["traceEvents"]
    pids = {e["pid"] for e in obj["traceEvents"]}
    # one pid per warp; instant events for div/bar
    assert any(e["name"].startswith("add.f32") for e in obj["traceEvents"])
    assert any(e["ph"] == "i" and "DIV_PUSH" in e["name"] for e in obj["traceEvents"])


def test_perfetto_emits_tc_tma_barrier_tracks(tmp_path):
    import json
    from gpusim.trace.recorder import Recorder
    from gpusim.viz.perfetto import save_perfetto
    r = Recorder()
    r.mma(cycle=0, warp_id=0, pc=0, precision="f16",
          shape_m=16, shape_n=8, shape_k=16, accum_dtype="f32",
          flops_count=4096)
    r.wgmma(kind="ISSUE", cycle=10, warp_group_id=0, pc=5,
             precision="f16", shape_m=64, shape_n=128, shape_k=16,
             completion_at=42)
    r.tma(cycle=20, completion_at=80, smem_dst=0, gmem_base=0,
          dim_x=8, dim_y=8, bytes_total=128, n_cache_lines=1,
          mbarrier_addr=0)
    r.mbarrier(kind="FLIP", cycle=85, cta_id=0, smem_addr=0)
    out = tmp_path / "trace.json"
    save_perfetto(r, out)
    data = json.loads(out.read_text())
    events = data.get("traceEvents", data) if isinstance(data, dict) else data
    pids = {e.get("pid", "") for e in events}
    assert any("TC" in str(p) or "tensor" in str(p).lower() for p in pids)
    assert any("TMA" in str(p) for p in pids)
    assert any("Barrier" in str(p) or "mbarrier" in str(p).lower() for p in pids)
