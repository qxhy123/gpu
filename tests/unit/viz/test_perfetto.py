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
