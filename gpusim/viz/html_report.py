from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
import plotly.graph_objects as go
import plotly.io as pio

from gpusim.trace.recorder import Recorder
from gpusim.analysis.stall import stall_breakdown, ipc_timeline
from gpusim.analysis.attribution import stall_by_source_line
from gpusim.analysis.metrics import bank_conflict_hist


_TPL_DIR = Path(__file__).parent
_env = Environment(loader=FileSystemLoader(_TPL_DIR), autoescape=select_autoescape())


def _total_cycles(rec: Recorder) -> int:
    """Return the max cycle seen across all Phase 3 events, or 1."""
    max_c = 1
    for e in rec.mma_events:
        max_c = max(max_c, e.cycle)
    for e in rec.wgmma_events:
        max_c = max(max_c, e.cycle)
    for e in rec.tma_events:
        max_c = max(max_c, e.completion_at)
    for e in rec.mbarrier_events:
        max_c = max(max_c, e.cycle)
    return max_c


def _render_tc_utilization(rec: Recorder) -> str:
    if not rec.mma_events and not rec.wgmma_events:
        return ""
    from gpusim.analysis.metrics import tc_utilization
    mma_df = pd.DataFrame([asdict(e) for e in rec.mma_events]) if rec.mma_events else pd.DataFrame()
    wgmma_df = pd.DataFrame([asdict(e) for e in rec.wgmma_events]) if rec.wgmma_events else pd.DataFrame()
    util = tc_utilization(mma_df, wgmma_df, total_cycles=_total_cycles(rec), n_sub_cores=4)
    return util.to_html(index=False)


def _render_precision_distribution(rec: Recorder) -> str:
    if not rec.mma_events and not rec.wgmma_events:
        return ""
    from gpusim.analysis.metrics import precision_distribution
    mma_df = pd.DataFrame([asdict(e) for e in rec.mma_events]) if rec.mma_events else pd.DataFrame()
    wgmma_df = pd.DataFrame([asdict(e) for e in rec.wgmma_events]) if rec.wgmma_events else pd.DataFrame()
    dist = precision_distribution(mma_df, wgmma_df)
    return dist.to_html()


def _render_wgmma_timeline(rec: Recorder) -> str:
    if not rec.wgmma_events:
        return ""
    df = pd.DataFrame([asdict(e) for e in rec.wgmma_events])
    return df.to_html(index=False)


def _render_mbarrier_table(rec: Recorder) -> str:
    if not rec.mbarrier_events:
        return ""
    df = pd.DataFrame([asdict(e) for e in rec.mbarrier_events])
    return df.to_html(index=False)


def _render_per_sm_utilization(rec, cycles):
    if not rec.cta_dispatch_events:
        return ""
    sm_ids = sorted({e.sm_id for e in rec.cta_dispatch_events})
    if not sm_ids:
        return ""
    n_sm = max(sm_ids) + 1
    warp_segments = list(rec.all_warp_segments())
    if not warp_segments:
        return pd.DataFrame({"sm_id": list(range(n_sm)),
                              "util": [0.0] * n_sm}).to_html(index=False)
    warp_state_df = pd.DataFrame([{"warp_id": s.warp_id, "start": s.start,
                                    "end": s.end, "state": s.state, "pc": s.pc}
                                   for s in warp_segments])
    from gpusim.analysis.metrics import per_sm_utilization
    df = per_sm_utilization(warp_state_df, cycles, n_sm)
    return df.to_html(index=False)


def _render_cta_dispatch(rec):
    if not rec.cta_dispatch_events:
        return ""
    df = pd.DataFrame([asdict(e) for e in rec.cta_dispatch_events])
    return df.to_html(index=False)


def _render_l2_mshr_pressure(rec, cycles):
    if not rec.l2_mshr_events:
        return ""
    df = pd.DataFrame([asdict(e) for e in rec.l2_mshr_events])
    return df.to_html(index=False)


def _render_bulk_store_table(rec):
    if not rec.bulk_store_events:
        return ""
    df = pd.DataFrame([asdict(e) for e in rec.bulk_store_events])
    return df.to_html(index=False)


def _ws_df(rec: Recorder) -> pd.DataFrame:
    segs = list(rec.all_warp_segments())
    return pd.DataFrame([{"warp_id":s.warp_id,"start":s.start,"end":s.end,
                          "state":s.state,"pc":s.pc} for s in segs])


def _issues_df(rec: Recorder) -> pd.DataFrame:
    evs = rec.instr_issues()
    return pd.DataFrame([{"cycle":e.cycle,"warp_id":e.warp_id,"pc":e.pc,
                          "op":e.op,"file":e.src_loc[0],"line":e.src_loc[1],
                          "active_mask":e.active_mask} for e in evs])


def _smem_df(rec: Recorder) -> pd.DataFrame:
    evs = rec.smem_accesses()
    return pd.DataFrame([{"cycle":e.cycle,"warp_id":e.warp_id,
                          "conflict_degree":e.conflict_degree} for e in evs])


def build_html(rec: Recorder, *, kernel_name: str, grid, block,
               occupancy: dict, cycles: int) -> str:
    ws = _ws_df(rec)
    issues = _issues_df(rec)
    smem = _smem_df(rec)

    sb = stall_breakdown(ws) if not ws.empty else {}
    ipc = ipc_timeline(ws) if not ws.empty else pd.Series(dtype=int)

    stall_pie = go.Figure([go.Pie(labels=list(sb.keys()), values=list(sb.values()))])
    ipc_line = go.Figure([go.Scatter(x=list(ipc.index), y=list(ipc.values), mode="lines")])

    stall_table = stall_by_source_line(issues_df=issues, warp_state_df=ws) \
        if not issues.empty and not ws.empty else pd.DataFrame()
    bank_hist = bank_conflict_hist(smem) if not smem.empty else pd.DataFrame()

    # Phase 2 additions
    from gpusim.viz.notebook import l1_events_dataframe, l2_events_dataframe, hbm_events_dataframe
    from gpusim.analysis.metrics import (
        cache_hierarchy_breakdown, channel_utilization, row_buffer_hit_rate,
        wb_traffic_fraction,
    )
    l1 = l1_events_dataframe(rec)
    l2 = l2_events_dataframe(rec)
    hbm = hbm_events_dataframe(rec)

    if not l1.empty:
        breakdown = cache_hierarchy_breakdown(l1, l2)
        cache_hierarchy_html = pd.DataFrame([breakdown]).to_html(index=False)
        cache_pie = go.Figure([go.Pie(labels=list(breakdown.keys()),
                                       values=list(breakdown.values()))])
        cache_pie_json = pio.to_json(cache_pie)
    else:
        cache_hierarchy_html = "<i>(no cache events)</i>"
        cache_pie_json = None

    if not hbm.empty:
        cu = channel_utilization(hbm, cycles)
        channel_util_chart = go.Figure([go.Bar(
            x=[f"ch{i}" for i in range(len(cu))],
            y=list(cu),
        )])
        channel_util_chart.update_layout(title="Channel utilization", yaxis_range=[0, 1])
        channel_util_json = pio.to_json(channel_util_chart)

        rh_count = (hbm["row_kind"] == "ROW_HIT").sum()
        rm_count = (hbm["row_kind"] == "ROW_MISS").sum()
        row_buffer_pie = go.Figure([go.Pie(labels=["ROW_HIT", "ROW_MISS"],
                                            values=[rh_count, rm_count])])
        row_buffer_json = pio.to_json(row_buffer_pie)

        line_bytes = 128
        read_bytes = (hbm["kind"] == "READ").sum() * line_bytes
        wb_bytes = (hbm["kind"] == "WRITE_BACK").sum() * line_bytes
        wb_metrics = {"read_bytes": int(read_bytes),
                      "wb_bytes": int(wb_bytes),
                      "wb_frac": wb_traffic_fraction(hbm)}
    else:
        channel_util_json = None
        row_buffer_json = None
        wb_metrics = {"read_bytes": 0, "wb_bytes": 0, "wb_frac": 0.0}

    return _env.get_template("_template.html.j2").render(
        kernel_name=kernel_name,
        grid=grid, block=block, cycles=cycles, occupancy=occupancy,
        stall_pie_json=pio.to_json(stall_pie),
        ipc_line_json=pio.to_json(ipc_line),
        stall_table_html=stall_table.to_html(index=False) if not stall_table.empty else "<i>(no data)</i>",
        bank_table_html=bank_hist.to_html() if not bank_hist.empty else "<i>(no data)</i>",
        cache_hierarchy_html=cache_hierarchy_html,
        cache_pie_json=cache_pie_json,
        channel_util_json=channel_util_json,
        row_buffer_json=row_buffer_json,
        wb_metrics=wb_metrics,
        tc_utilization_html=_render_tc_utilization(rec),
        precision_distribution_html=_render_precision_distribution(rec),
        wgmma_timeline_html=_render_wgmma_timeline(rec),
        mbarrier_table_html=_render_mbarrier_table(rec),
        per_sm_utilization_html=_render_per_sm_utilization(rec, cycles),
        cta_dispatch_html=_render_cta_dispatch(rec),
        l2_mshr_pressure_html=_render_l2_mshr_pressure(rec, cycles),
        bulk_store_table_html=_render_bulk_store_table(rec),
        cluster_timeline_html=_render_cluster_timeline(rec),
        dsmem_traffic_html=_render_dsmem_traffic(rec),
        atomic_contention_html=_render_atomic_contention(rec),
        cooperative_epilogue_html=_render_cooperative_epilogue(rec),
        stream_concurrency_html=_render_stream_concurrency(rec),
        per_stream_breakdown_html=_render_per_stream_breakdown(rec),
        priority_dispatch_html=_render_priority_dispatch(rec),
        event_timeline_html=_render_event_timeline(rec),
        l2_window_heatmap_html=_render_l2_window_heatmap(rec),
    )


def _render_cluster_timeline(rec):
    from dataclasses import asdict
    if not rec.cluster_dispatch_events and not rec.cluster_barrier_events:
        return ""
    import pandas as pd
    parts = []
    if rec.cluster_dispatch_events:
        df = pd.DataFrame([asdict(e) for e in rec.cluster_dispatch_events])
        parts.append("<h3>Cluster dispatches</h3>" + df.to_html(index=False))
    if rec.cluster_barrier_events:
        df = pd.DataFrame([asdict(e) for e in rec.cluster_barrier_events])
        parts.append("<h3>Cluster barrier events</h3>" + df.to_html(index=False))
    return "\n".join(parts)


def _render_dsmem_traffic(rec):
    if not rec.cluster_dispatch_events:
        return ""
    from dataclasses import asdict
    import pandas as pd
    from gpusim.analysis.metrics import dsmem_remote_access_rate
    events = rec.instr_issues()
    if not events:
        return ""
    instr_df = pd.DataFrame([asdict(e) for e in events])
    rate = dsmem_remote_access_rate(instr_df)
    return f"<p>dsmem remote access rate: <b>{rate*100:.1f}%</b></p>"


def _render_atomic_contention(rec):
    if not rec.atomic_events:
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.atomic_events])
    parts = []
    parts.append("<h3>Atomic events</h3>" + df.head(20).to_html(index=False))
    if "line_addr" in df.columns:
        per_line = df.groupby("line_addr").agg(
            count=("cycle", "size"),
            avg_latency=("latency", "mean"),
        ).reset_index().sort_values("count", ascending=False).head(10)
        parts.append("<h3>Hot lines (top 10)</h3>" + per_line.to_html(index=False))
    return "\n".join(parts)


def _render_cooperative_epilogue(rec):
    if not rec.bulk_store_events:
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.bulk_store_events])
    return "<h3>Cooperative epilogue (bulk store events)</h3>" + df.to_html(index=False)


def _render_priority_dispatch(rec):
    if not getattr(rec, "kernel_launch_events", None):
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.kernel_launch_events])
    return "<h3>Kernel launches by stream</h3>" + df.to_html(index=False)


def _render_event_timeline(rec):
    if not getattr(rec, "stream_event_events", None):
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.stream_event_events])
    return "<h3>Stream events timeline</h3>" + df.to_html(index=False)


def _render_l2_window_heatmap(rec):
    if not getattr(rec, "instr_events", None) and not getattr(rec, "instr_issues", None):
        return ""
    return "<h3>L2 access (placeholder for window heatmap)</h3>"


def _render_stream_concurrency(rec):
    if not getattr(rec, "kernel_launch_events", None):
        return ""
    from dataclasses import asdict
    import pandas as pd
    df = pd.DataFrame([asdict(e) for e in rec.kernel_launch_events])
    parts = []
    parts.append("<h3>Kernel launches by stream</h3>" + df.to_html(index=False))
    return "\n".join(parts)


def _render_per_stream_breakdown(rec):
    if not getattr(rec, "kernel_launch_events", None):
        return ""
    import pandas as pd
    rows = []
    streams = set()
    for ev_attr in ["instr_events", "atomic_events", "mma_events",
                     "bulk_load_events", "bulk_store_events"]:
        for e in getattr(rec, ev_attr, []) or []:
            sid = getattr(e, "stream_id", 0)
            streams.add(sid)
    if not streams:
        return ""
    for sid in sorted(streams):
        rows.append({
            "stream_id": sid,
            "instr_events": sum(1 for e in (getattr(rec, "instr_events", []) or [])
                                  if getattr(e, "stream_id", 0) == sid),
            "atomic_events": sum(1 for e in (getattr(rec, "atomic_events", []) or [])
                                   if getattr(e, "stream_id", 0) == sid),
            "memory_events": sum(1 for e in (getattr(rec, "instr_events", []) or [])
                                   if getattr(e, "stream_id", 0) == sid
                                   and ("ld" in getattr(e, "op", "")
                                        or "st" in getattr(e, "op", ""))),
        })
    if not rows:
        return ""
    return "<h3>Per-stream event breakdown</h3>" + pd.DataFrame(rows).to_html(index=False)


def save_html(rec: Recorder, path: str | Path, **kwargs) -> None:
    Path(path).write_text(build_html(rec, **kwargs))
