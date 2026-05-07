from __future__ import annotations
import pandas as pd


def _expand_segments(df: pd.DataFrame) -> pd.DataFrame:
    """Convert RLE segments into per-cycle rows: (cycle, warp_id, state, pc)."""
    cycles = (df["end"] - df["start"] + 1).astype(int)
    out = df.loc[df.index.repeat(cycles)].copy()
    out["cycle"] = out.groupby(level=0).cumcount() + out["start"].values
    return out[["cycle", "warp_id", "state", "pc"]].reset_index(drop=True)


def stall_breakdown(warp_state_df: pd.DataFrame) -> dict[str, int]:
    df = warp_state_df.copy()
    df["cycles"] = df["end"] - df["start"] + 1
    grp = df.groupby("state")["cycles"].sum()
    return {k: int(v) for k, v in grp.items()}


def ipc_timeline(warp_state_df: pd.DataFrame) -> pd.Series:
    issued = warp_state_df[warp_state_df["state"] == "ISSUED"]
    rows = []
    for _, r in issued.iterrows():
        for c in range(int(r["start"]), int(r["end"]) + 1):
            rows.append(c)
    if not rows:
        return pd.Series(dtype=int)
    return pd.Series(rows).value_counts().sort_index()
