import pandas as pd
from gpusim.analysis.metrics import (
    bank_conflict_hist, coalescing_per_instr,
    divergence_cost, occupancy_timeline,
)


def test_bank_conflict_hist_counts_per_pc():
    smem = pd.DataFrame([
        {"cycle": 0, "warp_id": 0, "conflict_degree": 1},
        {"cycle": 5, "warp_id": 0, "conflict_degree": 4},
        {"cycle": 6, "warp_id": 1, "conflict_degree": 1},
        {"cycle": 7, "warp_id": 1, "conflict_degree": 4},
    ])
    h = bank_conflict_hist(smem)
    assert h.loc[1, "count"] == 2
    assert h.loc[4, "count"] == 2


def test_coalescing_per_instr_groups_by_pc():
    issues = pd.DataFrame([
        {"cycle": 0, "pc": 3, "op": "ld.global.f32", "line": 7},
        {"cycle": 1, "pc": 3, "op": "ld.global.f32", "line": 7},
    ])
    gmem = pd.DataFrame([
        {"cycle": 0, "warp_id": 0, "n_transactions": 1, "efficiency": 1.0},
        {"cycle": 1, "warp_id": 1, "n_transactions": 2, "efficiency": 0.5},
    ])
    out = coalescing_per_instr(issues, gmem)
    assert len(out) >= 1
    # average efficiency for pc=3 is 0.75
    row = out[out["pc"] == 3].iloc[0]
    assert abs(row["efficiency_mean"] - 0.75) < 1e-9
    assert row["n_transactions_mean"] == 1.5


def test_divergence_cost_sums_serial_state_cycles():
    states = pd.DataFrame([
        {"warp_id": 0, "start": 0, "end": 4, "state": "DIVERGENCE_SERIAL", "pc": 0},
        {"warp_id": 0, "start": 5, "end": 9, "state": "ISSUED", "pc": 0},
    ])
    assert divergence_cost(states) == 5


def test_occupancy_timeline_counts_active_warps():
    states = pd.DataFrame([
        {"warp_id": 0, "start": 0, "end": 5, "state": "ISSUED", "pc": 0},
        {"warp_id": 0, "start": 6, "end": 10, "state": "IDLE", "pc": -1},
        {"warp_id": 1, "start": 3, "end": 8, "state": "ISSUED", "pc": 0},
        {"warp_id": 1, "start": 9, "end": 10, "state": "IDLE", "pc": -1},
    ])
    s = occupancy_timeline(states)
    # at cycle 0: warp0 active, warp1 not yet -> 1 active
    assert s.loc[0] == 1
    assert s.loc[5] == 2  # both active
    assert s.loc[10] == 0
