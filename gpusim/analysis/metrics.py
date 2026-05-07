from __future__ import annotations
import pandas as pd


def bank_conflict_hist(smem_df: pd.DataFrame) -> pd.DataFrame:
    g = smem_df.groupby("conflict_degree").size().rename("count").reset_index()
    return g.set_index("conflict_degree")


def coalescing_per_instr(issues_df: pd.DataFrame,
                         gmem_df: pd.DataFrame) -> pd.DataFrame:
    if issues_df.empty or gmem_df.empty:
        return pd.DataFrame(columns=["pc", "efficiency_mean", "n_transactions_mean", "count"])
    # join by cycle (the gmem event happens in the same cycle as the issue)
    joined = gmem_df.merge(issues_df[["cycle", "pc", "op", "line"]], on="cycle", how="left")
    out = (joined.groupby(["pc"])
           .agg(efficiency_mean=("efficiency", "mean"),
                n_transactions_mean=("n_transactions", "mean"),
                count=("efficiency", "count"))
           .reset_index())
    return out


def divergence_cost(warp_state_df: pd.DataFrame) -> int:
    df = warp_state_df[warp_state_df["state"] == "DIVERGENCE_SERIAL"]
    if df.empty:
        return 0
    return int((df["end"] - df["start"] + 1).sum())


def occupancy_timeline(warp_state_df: pd.DataFrame) -> pd.Series:
    """Per-cycle count of warps whose state is not IDLE."""
    if warp_state_df.empty:
        return pd.Series(dtype=int)
    max_cycle = int(warp_state_df["end"].max())
    counts = [0] * (max_cycle + 1)
    for _, r in warp_state_df.iterrows():
        if r["state"] == "IDLE":
            continue
        for c in range(int(r["start"]), int(r["end"]) + 1):
            counts[c] += 1
    return pd.Series(counts)
