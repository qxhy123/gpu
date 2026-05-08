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


def l1_hit_rate(l1_df: pd.DataFrame) -> float:
    if l1_df.empty:
        return 0.0
    hits = (l1_df["kind"] == "HIT").sum()
    total = len(l1_df)
    return hits / total if total > 0 else 0.0


def l2_hit_rate(l2_df: pd.DataFrame) -> float:
    if l2_df.empty:
        return 0.0
    access_kinds = ["HIT", "MISS_LOAD", "MISS_STORE"]
    accesses = l2_df[l2_df["kind"].isin(access_kinds)]
    if accesses.empty:
        return 0.0
    hits = (accesses["kind"] == "HIT").sum()
    return hits / len(accesses)


def mshr_merge_rate(l1_df: pd.DataFrame) -> float:
    if l1_df.empty:
        return 0.0
    misses = l1_df[l1_df["kind"].isin(["MISS_NEW", "MISS_MERGE"])]
    if misses.empty:
        return 0.0
    merges = (misses["kind"] == "MISS_MERGE").sum()
    return merges / len(misses)


def cache_hierarchy_breakdown(l1_df: pd.DataFrame,
                               l2_df: pd.DataFrame) -> dict[str, float]:
    """Returns fractions of total memory traffic that hit each level."""
    if l1_df.empty:
        return {"l1_hit": 0.0, "l2_hit": 0.0, "hbm": 0.0}
    total = len(l1_df)
    l1_hit = (l1_df["kind"] == "HIT").sum()
    l1_misses = (l1_df["kind"] == "MISS_NEW").sum()  # only NEW, not MERGE
    if l1_misses > 0 and not l2_df.empty:
        l2_hit_count = (l2_df["kind"] == "HIT").sum()
        # the L1 misses that hit L2 (capped by l1_misses)
        l2_hit_count = min(l2_hit_count, l1_misses)
    else:
        l2_hit_count = 0
    hbm_count = max(0, l1_misses - l2_hit_count)
    return {
        "l1_hit": l1_hit / total,
        "l2_hit": l2_hit_count / total,
        "hbm": hbm_count / total,
    }


def bandwidth_per_channel(hbm_df: pd.DataFrame, total_cycles: int,
                           line_bytes: int = 128, n_channels: int = 8) -> pd.Series:
    """Bytes per cycle per channel."""
    out = [0.0] * n_channels
    if hbm_df.empty or total_cycles == 0:
        return pd.Series(out)
    counts = hbm_df.groupby("channel").size()
    for c, count in counts.items():
        out[c] = count * line_bytes / total_cycles
    return pd.Series(out)


def channel_utilization(hbm_df: pd.DataFrame, total_cycles: int,
                         n_channels: int = 8) -> pd.Series:
    """Fraction of cycles each channel was busy serving requests."""
    out = [0.0] * n_channels
    if hbm_df.empty or total_cycles == 0:
        return pd.Series(out)
    # Treat each request as occupying (served_at - cycle - queue_wait) cycles
    busy_per_chan = [0] * n_channels
    for _, r in hbm_df.iterrows():
        c = int(r["channel"])
        busy = int(r["served_at"]) - (int(r["cycle"]) + int(r["queue_wait"]))
        busy_per_chan[c] += busy
    for c in range(n_channels):
        out[c] = min(1.0, busy_per_chan[c] / total_cycles)
    return pd.Series(out)


def row_buffer_hit_rate(hbm_df: pd.DataFrame) -> float:
    if hbm_df.empty:
        return 0.0
    hits = (hbm_df["row_kind"] == "ROW_HIT").sum()
    return hits / len(hbm_df)


def queue_wait_distribution(hbm_df: pd.DataFrame) -> pd.Series:
    """Histogram of queue_wait values."""
    if hbm_df.empty:
        return pd.Series(dtype=int)
    return hbm_df["queue_wait"].value_counts().sort_index()


def wb_traffic_fraction(hbm_df: pd.DataFrame) -> float:
    if hbm_df.empty:
        return 0.0
    wb = (hbm_df["kind"] == "WRITE_BACK").sum()
    return wb / len(hbm_df)
