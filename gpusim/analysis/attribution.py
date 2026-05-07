from __future__ import annotations
import pandas as pd


def stall_by_source_line(*, issues_df: pd.DataFrame,
                         warp_state_df: pd.DataFrame) -> pd.DataFrame:
    """For each (warp_id, pc), the warp may dwell in non-ISSUED states
    while waiting to issue that pc. Attribute those cycles to the (file, line)
    of that pc.
    """
    pc_to_loc = (issues_df.groupby("pc")
                 .agg({"file": "first", "line": "first", "op": "first"})
                 .reset_index())
    df = warp_state_df.copy()
    df["cycles"] = df["end"] - df["start"] + 1
    merged = df.merge(pc_to_loc, on="pc", how="left")
    grouped = (merged.groupby(["file", "line", "op", "state"])["cycles"]
               .sum().reset_index())
    return grouped
