import pandas as pd
from gpusim.analysis.attribution import stall_by_source_line


def test_attribution_groups_stall_by_pc_then_src_line():
    issues = pd.DataFrame([
        {"cycle": 0, "warp_id": 0, "pc": 0, "op": "add.f32", "file": "k.ptx", "line": 5, "active_mask": 0xFFFFFFFF},
        {"cycle": 4, "warp_id": 0, "pc": 1, "op": "add.f32", "file": "k.ptx", "line": 6, "active_mask": 0xFFFFFFFF},
    ])
    states = pd.DataFrame([
        {"warp_id": 0, "start": 0, "end": 0, "state": "ISSUED", "pc": 0},
        {"warp_id": 0, "start": 1, "end": 3, "state": "SCOREBOARD", "pc": 1},
        {"warp_id": 0, "start": 4, "end": 4, "state": "ISSUED", "pc": 1},
    ])
    df = stall_by_source_line(issues_df=issues, warp_state_df=states)
    # line 6 had 3 cycles of SCOREBOARD attributed to it
    row = df[(df["line"] == 6) & (df["state"] == "SCOREBOARD")].iloc[0]
    assert row["cycles"] == 3
