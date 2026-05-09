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

    # Phase 3: TC track (per warp-group / per warp)
    for ev in rec.wgmma_events:
        if ev.kind == "ISSUE":
            events.append({
                "name": f"wgmma {ev.precision}",
                "cat": "tc",
                "ph": "X",
                "ts": ev.cycle,
                "dur": max(1, ev.completion_at - ev.cycle),
                "pid": f"TC_wg{ev.warp_group_id}",
                "tid": "wgmma",
                "args": {"shape": f"m{ev.shape_m}n{ev.shape_n}k{ev.shape_k}"},
            })
        elif ev.kind == "WAIT_GROUP":
            events.append({
                "name": "wait_group",
                "cat": "tc",
                "ph": "i",
                "ts": ev.cycle,
                "pid": f"TC_wg{ev.warp_group_id}",
                "tid": "wgmma",
            })

    for ev in rec.mma_events:
        events.append({
            "name": f"mma {ev.precision}",
            "cat": "tc",
            "ph": "i",
            "ts": ev.cycle,
            "pid": f"TC_w{ev.warp_id}",
            "tid": "mma",
        })

    # TMA track (per CTA — TmaEvent doesn't store cta_id; use placeholder)
    for ev in rec.tma_events:
        events.append({
            "name": "tma_copy",
            "cat": "tma",
            "ph": "X",
            "ts": ev.cycle,
            "dur": max(1, ev.completion_at - ev.cycle),
            "pid": "TMA_cta_unknown",
            "tid": "tma",
            "args": {"bytes": ev.bytes_total},
        })

    # Mbarrier flip track (per CTA)
    for ev in rec.mbarrier_events:
        if ev.kind == "FLIP":
            events.append({
                "name": "flip",
                "cat": "barrier",
                "ph": "i",
                "ts": ev.cycle,
                "pid": f"Barrier_cta{ev.cta_id}",
                "tid": "mbar",
                "args": {"phase": ev.phase},
            })

    # Phase 4: per-SM CTA dispatch instants
    for ev in rec.cta_dispatch_events:
        events.append({
            "name": f"CTA {ev.cta_id}",
            "cat": "cta", "ph": "i", "ts": ev.cycle,
            "pid": f"SM{ev.sm_id}", "tid": "cta_dispatch",
        })

    # L2 MSHR events
    for ev in rec.l2_mshr_events:
        events.append({
            "name": f"L2 {ev.kind}",
            "cat": "l2_mshr", "ph": "i", "ts": ev.cycle,
            "pid": "L2_MSHR", "tid": ev.kind.lower(),
            "args": {"line_addr": ev.line_addr, "sm_id": ev.sm_id,
                     "n_waiters": ev.n_waiters},
        })

    # BulkStore in-flight as duration events per warp-group
    for ev in rec.bulk_store_events:
        if ev.kind == "ISSUE":
            events.append({
                "name": "bulk_store",
                "cat": "tma_store", "ph": "X", "ts": ev.cycle,
                "dur": max(1, ev.completion_at - ev.cycle),
                "pid": f"TMA_Store_wg{ev.warp_group_id}", "tid": "bulk",
                "args": {"bytes": ev.bytes_total, "sm_id": ev.sm_id},
            })
        elif ev.kind == "WAIT_GROUP":
            events.append({
                "name": "wait_group",
                "cat": "tma_store", "ph": "i", "ts": ev.cycle,
                "pid": f"TMA_Store_wg{ev.warp_group_id}", "tid": "wait",
                "args": {"sm_id": ev.sm_id, "wait_n": ev.wait_n},
            })

    # Phase 5: cluster dispatch instants
    for ev in rec.cluster_dispatch_events:
        events.append({
            "name": f"Cluster {ev.cluster_id} dispatched",
            "cat": "cluster", "ph": "i", "ts": ev.cycle,
            "pid": f"Cluster{ev.cluster_id}", "tid": "dispatch",
            "args": {"sm_ids": list(ev.sm_ids), "cta_ids": list(ev.cta_ids)},
        })

    # Cluster barrier events
    for ev in rec.cluster_barrier_events:
        events.append({
            "name": f"barrier.cluster.{ev.kind.lower()}",
            "cat": "cluster_barrier", "ph": "i", "ts": ev.cycle,
            "pid": f"Cluster{ev.cluster_id}", "tid": "barrier",
            "args": {"cta_id": ev.cta_id, "rank": ev.rank, "sm_id": ev.sm_id,
                     "arrived_count": ev.arrived_count},
        })

    # Phase 6 atomic events
    for ev in rec.atomic_events:
        events.append({
            "name": f"{ev.kind.lower()}.{ev.space}.{ev.op}",
            "cat": "atomic", "ph": "X", "ts": ev.cycle,
            "dur": max(1, ev.latency),
            "pid": "Atomic", "tid": ev.kind.lower(),
            "args": {"line_addr": ev.line_addr, "sm_id": ev.sm_id,
                     "n_lanes": ev.n_lanes,
                     "queue_depth_before": ev.queue_depth_before},
        })

    # Phase 7 stream swimlanes — one per kernel_launch event
    for ev in getattr(rec, "kernel_launch_events", []):
        events.append({
            "name": ev.kernel_name,
            "cat": "kernel_launch", "ph": "X",
            "ts": ev.launch_cycle,
            "dur": max(1, ev.complete_cycle - ev.launch_cycle),
            "pid": f"Stream-{ev.stream_id}",
            "tid": "kernel",
            "args": {
                "stream_id": ev.stream_id,
                "n_ctas": ev.n_ctas,
                "grid": list(ev.grid),
                "block": list(ev.block),
            },
        })

    # Phase 8 stream events as instant events
    for ev in getattr(rec, "stream_event_events", []):
        events.append({
            "name": f"event_{ev.op}_{ev.event_id}",
            "cat": "stream_event", "ph": "i",
            "ts": ev.cycle, "s": "g",
            "pid": f"Stream-{ev.stream_id}",
            "tid": "events",
            "args": {"event_id": ev.event_id, "op": ev.op},
        })

    return {"traceEvents": events, "displayTimeUnit": "ns"}


def save_perfetto(rec: Recorder, path: str | Path) -> None:
    Path(path).write_text(json.dumps(build_perfetto(rec)))
