from __future__ import annotations
import json
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
    )


def save_html(rec: Recorder, path: str | Path, **kwargs) -> None:
    Path(path).write_text(build_html(rec, **kwargs))
