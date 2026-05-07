from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from gpusim.trace.recorder import Recorder


def warp_state_dataframe(rec: Recorder) -> pd.DataFrame:
    segs = list(rec.all_warp_segments())
    return pd.DataFrame([{"warp_id":s.warp_id,"start":s.start,"end":s.end,
                          "state":s.state,"pc":s.pc} for s in segs])


def stall_dataframe(rec: Recorder) -> pd.DataFrame:
    df = warp_state_dataframe(rec)
    if df.empty: return pd.DataFrame(columns=["state","cycles"])
    df["cycles"] = df["end"] - df["start"] + 1
    return df.groupby("state")["cycles"].sum().reset_index()


def warp_timeline_figure(rec: Recorder, warp_id: int) -> go.Figure:
    segs = [s for s in rec.all_warp_segments() if s.warp_id == warp_id]
    fig = go.Figure()
    for s in segs:
        fig.add_trace(go.Bar(
            x=[s.end - s.start + 1], y=[f"warp{warp_id}"],
            base=s.start, orientation="h",
            name=s.state, hovertext=f"pc={s.pc} {s.start}-{s.end}",
        ))
    fig.update_layout(barmode="stack", title=f"Warp {warp_id} timeline",
                      xaxis_title="cycle")
    return fig
