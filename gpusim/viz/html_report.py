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

    return _env.get_template("_template.html.j2").render(
        kernel_name=kernel_name,
        grid=grid, block=block, cycles=cycles, occupancy=occupancy,
        stall_pie_json=pio.to_json(stall_pie),
        ipc_line_json=pio.to_json(ipc_line),
        stall_table_html=stall_table.to_html(index=False) if not stall_table.empty else "<i>(no data)</i>",
        bank_table_html=bank_hist.to_html() if not bank_hist.empty else "<i>(no data)</i>",
    )


def save_html(rec: Recorder, path: str | Path, **kwargs) -> None:
    Path(path).write_text(build_html(rec, **kwargs))
