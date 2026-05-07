from __future__ import annotations
import json
from pathlib import Path
from gpusim.trace.recorder import Recorder


def build_perfetto(rec: Recorder) -> dict:
    events = []
    # one process per warp
    warps = set()
    for s in rec.all_warp_segments():
        warps.add(s.warp_id)

    for w in sorted(warps):
        events.append({"name":"process_name","ph":"M","pid":w,"tid":0,
                       "args":{"name":f"warp{w}"}})

    # WARP_STATE → "X" complete events (each segment becomes one slice)
    for s in rec.all_warp_segments():
        events.append({
            "name": s.state, "ph": "X", "pid": s.warp_id, "tid": 0,
            "ts": s.start, "dur": s.end - s.start + 1,
            "args": {"pc": s.pc},
        })

    # instr_issue as instant events
    for e in rec.instr_issues():
        events.append({
            "name": f"{e.op} pc={e.pc}", "ph": "i", "pid": e.warp_id, "tid": 1,
            "ts": e.cycle, "s": "t",
            "args": {"line": e.src_loc[1], "active_mask": hex(e.active_mask)},
        })

    # divergence pushes/pops as instant
    for e in rec.div_events():
        events.append({
            "name": f"DIV_{e.kind} pc={e.pc}", "ph": "i", "pid": e.warp_id, "tid": 2,
            "ts": e.cycle, "s": "t",
            "args": {"rpc": e.rpc, "taken_mask": hex(e.taken_mask)},
        })

    for e in rec.bar_events():
        events.append({
            "name": f"BAR_{e.kind}", "ph": "i", "pid": -1, "tid": e.cta_id,
            "ts": e.cycle, "s": "g",
            "args": {"cta_id": e.cta_id, "barrier_id": e.barrier_id},
        })

    return {"traceEvents": events, "displayTimeUnit": "ns"}


def save_perfetto(rec: Recorder, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_perfetto(rec)))
