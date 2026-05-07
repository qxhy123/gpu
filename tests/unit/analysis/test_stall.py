import pandas as pd
from gpusim.analysis.stall import stall_breakdown, ipc_timeline


def test_stall_breakdown_counts_state_cycles():
    # warp 0: 0..4 ISSUED (5), 5..7 SCOREBOARD (3)
    # warp 1: 0..7 IDLE (8)
    df = pd.DataFrame([
        {"warp_id": 0, "start": 0, "end": 4, "state": "ISSUED", "pc": 0},
        {"warp_id": 0, "start": 5, "end": 7, "state": "SCOREBOARD", "pc": 1},
        {"warp_id": 1, "start": 0, "end": 7, "state": "IDLE", "pc": -1},
    ])
    out = stall_breakdown(df)
    assert out["ISSUED"] == 5
    assert out["SCOREBOARD"] == 3
    assert out["IDLE"] == 8


def test_ipc_timeline_counts_issuances_per_cycle():
    df = pd.DataFrame([
        {"warp_id": 0, "start": 0, "end": 2, "state": "ISSUED", "pc": 0},   # 3 issues at c=0,1,2
        {"warp_id": 1, "start": 1, "end": 1, "state": "ISSUED", "pc": 0},   # 1 issue at c=1
    ])
    series = ipc_timeline(df)
    assert series.loc[0] == 1
    assert series.loc[1] == 2
    assert series.loc[2] == 1
