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


def l1_events_dataframe(rec) -> pd.DataFrame:
    evs = rec.l1_accesses()
    return pd.DataFrame([{"cycle":e.cycle, "warp_id":e.warp_id, "kind":e.kind,
                          "line_addr":e.line_addr, "set_idx":e.set_idx,
                          "way":e.way,
                          "mshr_slot": e.mshr_slot if e.mshr_slot is not None else -1}
                         for e in evs])


def l2_events_dataframe(rec) -> pd.DataFrame:
    evs = rec.l2_accesses()
    return pd.DataFrame([{"cycle":e.cycle, "kind":e.kind,
                          "line_addr":e.line_addr, "set_idx":e.set_idx,
                          "way":e.way, "victim_addr":e.victim_addr}
                         for e in evs])


def hbm_events_dataframe(rec) -> pd.DataFrame:
    evs = rec.hbm_accesses()
    return pd.DataFrame([{"cycle":e.cycle, "served_at":e.served_at,
                          "addr":e.addr, "channel":e.channel,
                          "bank":e.bank, "row":e.row,
                          "kind":e.kind, "row_kind":e.row_kind,
                          "queue_wait":e.queue_wait}
                         for e in evs])


def mma_events_dataframe(rec) -> pd.DataFrame:
    from dataclasses import asdict
    if not rec.mma_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.mma_events])


def wgmma_events_dataframe(rec) -> pd.DataFrame:
    from dataclasses import asdict
    if not rec.wgmma_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.wgmma_events])


def tma_events_dataframe(rec) -> pd.DataFrame:
    from dataclasses import asdict
    if not rec.tma_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.tma_events])


def mbarrier_events_dataframe(rec) -> pd.DataFrame:
    from dataclasses import asdict
    if not rec.mbarrier_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.mbarrier_events])


def cta_dispatch_events_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    if not rec.cta_dispatch_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.cta_dispatch_events])


def l2_mshr_events_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    if not rec.l2_mshr_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.l2_mshr_events])


def bulk_store_events_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    if not rec.bulk_store_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.bulk_store_events])


def cluster_dispatch_events_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    if not rec.cluster_dispatch_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.cluster_dispatch_events])


def cluster_barrier_events_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    if not rec.cluster_barrier_events:
        return pd.DataFrame()
    return pd.DataFrame([asdict(e) for e in rec.cluster_barrier_events])


def instr_issue_dataframe(rec):
    import pandas as pd
    from dataclasses import asdict
    events = getattr(rec, "instr_issue_events", None) or getattr(rec, "instr_issues", None)
    if not events:
        return pd.DataFrame(columns=["op"])
    return pd.DataFrame([asdict(e) for e in events])
