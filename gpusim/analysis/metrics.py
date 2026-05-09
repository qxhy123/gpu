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


def tc_utilization(mma_df, wgmma_df, total_cycles: int,
                    n_sub_cores: int = 4) -> "pd.DataFrame":
    """Per-sub-core TC busy %."""
    busy = [0] * n_sub_cores
    if mma_df is not None and not mma_df.empty:
        for _, r in mma_df.iterrows():
            sc = int(r["warp_id"]) % n_sub_cores
            busy[sc] += 1
    if wgmma_df is not None and not wgmma_df.empty:
        for _, r in wgmma_df[wgmma_df["kind"] == "ISSUE"].iterrows():
            sc = int(r["warp_group_id"]) % n_sub_cores
            busy[sc] += 4
    util = [b / max(total_cycles, 1) for b in busy]
    return pd.DataFrame({f"sub_core_{i}": [util[i]] for i in range(n_sub_cores)})


def precision_distribution(mma_df, wgmma_df) -> "pd.DataFrame":
    rows: list[dict] = []
    if mma_df is not None and not mma_df.empty:
        for _, r in mma_df.iterrows():
            rows.append({"precision": r["precision"], "flops": int(r["flops_count"])})
    if wgmma_df is not None and not wgmma_df.empty:
        for _, r in wgmma_df[wgmma_df["kind"] == "ISSUE"].iterrows():
            flops = 2 * int(r["shape_m"]) * int(r["shape_n"]) * int(r["shape_k"])
            rows.append({"precision": r["precision"], "flops": flops})
    if not rows:
        return pd.DataFrame(columns=["count", "flops"])
    df = pd.DataFrame(rows)
    out = df.groupby("precision").agg(count=("flops", "size"), flops=("flops", "sum"))
    return out


def effective_tflops(mma_df, wgmma_df, total_cycles: int,
                       freq_ghz: float = 1.0) -> dict:
    sec = total_cycles / (freq_ghz * 1e9)
    if sec <= 0:
        return {}
    out: dict[str, float] = {}
    if mma_df is not None and not mma_df.empty:
        for prec, grp in mma_df.groupby("precision"):
            out[prec] = float(grp["flops_count"].sum()) / sec / 1e12
    if wgmma_df is not None and not wgmma_df.empty:
        for prec, grp in wgmma_df[wgmma_df["kind"] == "ISSUE"].groupby("precision"):
            flops = (2 * grp["shape_m"] * grp["shape_n"] * grp["shape_k"]).sum()
            out[prec] = out.get(prec, 0.0) + float(flops) / sec / 1e12
    return out


def async_overlap_ratio(wgmma_df, warp_state_df) -> float:
    """Fraction of in-flight wgmma cycles during which the issuing warp was not WGMMA_WAIT."""
    if wgmma_df is None or wgmma_df.empty:
        return 0.0
    issues = wgmma_df[wgmma_df["kind"] == "ISSUE"]
    if issues.empty:
        return 0.0
    total_inflight = 0
    overlapped = 0
    for _, row in issues.iterrows():
        start = int(row["cycle"])
        end = int(row["completion_at"])
        total_inflight += max(0, end - start)
        if warp_state_df is not None and not warp_state_df.empty:
            for _, ws in warp_state_df.iterrows():
                ws_start = max(start, int(ws["start"]))
                ws_end = min(end, int(ws["end"]))
                if ws_end > ws_start and ws.get("state") not in ("WGMMA_WAIT", "IDLE"):
                    overlapped += ws_end - ws_start
    return overlapped / max(total_inflight, 1)


def mbarrier_wait_distribution(wgmma_df, mbarrier_df) -> "pd.Series":
    """Histogram of WAIT_GROUP -> next FLIP duration."""
    if wgmma_df is None or wgmma_df.empty:
        return pd.Series(dtype=int)
    waits = wgmma_df[wgmma_df["kind"] == "WAIT_GROUP"]
    if waits.empty or mbarrier_df is None or mbarrier_df.empty:
        return pd.Series(dtype=int)
    flips = mbarrier_df[mbarrier_df["kind"] == "FLIP"]["cycle"].sort_values().tolist()
    durations: list[int] = []
    for _, row in waits.iterrows():
        wcycle = int(row["cycle"])
        next_flip = next((f for f in flips if f >= wcycle), wcycle)
        durations.append(next_flip - wcycle)
    return pd.Series(durations).value_counts().sort_index()


def wgmma_queue_pressure(wgmma_df, total_cycles: int) -> "pd.Series":
    """In-flight wgmma count per cycle."""
    pressure = [0] * (total_cycles + 1)
    if wgmma_df is None or wgmma_df.empty:
        return pd.Series(pressure)
    for _, row in wgmma_df[wgmma_df["kind"] == "ISSUE"].iterrows():
        s = int(row["cycle"])
        e = min(int(row["completion_at"]), total_cycles)
        for c in range(s, e + 1):
            pressure[c] += 1
    return pd.Series(pressure)


def tma_bandwidth_utilization(tma_df, total_cycles: int,
                                total_hbm_bw: float) -> float:
    """TMA bytes transferred as fraction of total HBM bandwidth capacity."""
    if tma_df is None or tma_df.empty or total_cycles <= 0 or total_hbm_bw <= 0:
        return 0.0
    total_bytes = float(tma_df["bytes_total"].sum())
    return total_bytes / total_hbm_bw


def per_sm_utilization(warp_state_df, total_cycles: int,
                         n_sm: int) -> "pd.DataFrame":
    busy = [0] * n_sm
    if warp_state_df is not None and not warp_state_df.empty:
        for _, r in warp_state_df.iterrows():
            sm_id = int(r.get("sm_id", -1))
            if sm_id < 0 or sm_id >= n_sm:
                continue
            state = r.get("state", "")
            if state in ("ISSUED", "DIVERGENCE_SERIAL"):
                busy[sm_id] += int(r["end"]) - int(r["start"]) + 1
    util = [b / max(total_cycles, 1) for b in busy]
    return pd.DataFrame({f"sm_{i}": [util[i]] for i in range(n_sm)})


def cta_to_sm_mapping(dispatch_df) -> "pd.DataFrame":
    if dispatch_df is None or dispatch_df.empty:
        return pd.DataFrame(columns=["cta_id", "sm_id", "dispatch_cycle"])
    out = dispatch_df.rename(columns={"cycle": "dispatch_cycle"})[
        ["cta_id", "sm_id", "dispatch_cycle"]]
    return out.sort_values("cta_id").reset_index(drop=True)


def cta_dispatch_latency(dispatch_df, cta_launch_df) -> "pd.Series":
    if dispatch_df is None or dispatch_df.empty:
        return pd.Series(dtype=int)
    if cta_launch_df is None or (hasattr(cta_launch_df, "empty") and cta_launch_df.empty):
        return dispatch_df["cycle"].value_counts().sort_index()
    merged = dispatch_df.merge(cta_launch_df, on="cta_id",
                                  suffixes=("_dispatch", "_launch"))
    durations = merged["cycle_dispatch"] - merged["cycle_launch"]
    return durations.value_counts().sort_index()


def l2_cross_sm_hit_rate(l2_events_df) -> float:
    if l2_events_df is None or l2_events_df.empty:
        return 0.0
    # origin_sm / hit_sm columns are only present when the L2 cache records
    # cross-SM provenance; fall back to 0.0 if they are absent.
    if "origin_sm" not in l2_events_df.columns or "hit_sm" not in l2_events_df.columns:
        return 0.0
    hits = l2_events_df[l2_events_df["kind"] == "HIT"]
    if hits.empty:
        return 0.0
    cross = (hits["origin_sm"] != hits["hit_sm"]).sum()
    return float(cross) / len(hits)


def l2_mshr_pressure(l2_mshr_events_df, total_cycles: int) -> "pd.Series":
    pressure = [0] * (total_cycles + 1)
    if l2_mshr_events_df is None or l2_mshr_events_df.empty:
        return pd.Series(pressure)
    in_flight: dict[int, int] = {}
    events = l2_mshr_events_df.sort_values("cycle")
    cycle = 0
    for _, row in events.iterrows():
        c = int(row["cycle"])
        for cy in range(cycle, min(c, total_cycles) + 1):
            pressure[cy] = len(in_flight)
        cycle = c
        line = int(row["line_addr"])
        if row["kind"] == "ALLOC":
            in_flight[line] = c
        elif row["kind"] == "RELEASE":
            in_flight.pop(line, None)
    for cy in range(cycle, total_cycles + 1):
        pressure[cy] = len(in_flight)
    return pd.Series(pressure)


def cluster_dispatch_latency(cluster_dispatch_df, cta_launch_df) -> "pd.Series":
    """Distribution of cluster dispatch cycle delays."""
    if cluster_dispatch_df is None or cluster_dispatch_df.empty:
        return pd.Series(dtype=int)
    return cluster_dispatch_df["cycle"].value_counts().sort_index()


def cluster_barrier_wait_distribution(cluster_barrier_df) -> "pd.Series":
    """For each cluster, compute cycles between first ARRIVE and WAIT_RELEASE."""
    if cluster_barrier_df is None or cluster_barrier_df.empty:
        return pd.Series(dtype=int)
    durations: list[int] = []
    for cluster_id, grp in cluster_barrier_df.groupby("cluster_id"):
        arrives = grp[grp["kind"] == "ARRIVE"]["cycle"]
        releases = grp[grp["kind"] == "WAIT_RELEASE"]["cycle"]
        if not arrives.empty and not releases.empty:
            durations.append(int(releases.min() - arrives.min()))
    return pd.Series(durations).value_counts().sort_index()


def dsmem_remote_access_rate(instr_issue_df) -> float:
    """Fraction of ld/st.shared.* ops that target cluster scope."""
    if instr_issue_df is None or instr_issue_df.empty:
        return 0.0
    shared_ops = instr_issue_df[instr_issue_df["op"].str.contains(r"\.shared")]
    if shared_ops.empty:
        return 0.0
    cluster_ops = shared_ops[shared_ops["op"].str.contains("shared::cluster")]
    return float(len(cluster_ops)) / len(shared_ops)


def atomic_throughput_per_line(atomic_df, total_cycles: int) -> "pd.DataFrame":
    """Per-line atomic throughput (count + atomic ops per cycle)."""
    if atomic_df is None or atomic_df.empty:
        return pd.DataFrame(columns=["line_addr", "atomic_count", "throughput"])
    grouped = atomic_df.groupby("line_addr").size().reset_index(name="atomic_count")
    grouped["throughput"] = grouped["atomic_count"] / max(total_cycles, 1)
    return grouped.sort_values("atomic_count", ascending=False)


def atomic_serialization_overhead(atomic_df, total_cycles: int) -> float:
    """Total atomic latency / total cycles (proxy for L2 ALU utilization)."""
    if atomic_df is None or atomic_df.empty or total_cycles <= 0:
        return 0.0
    total_latency = float(atomic_df["latency"].sum())
    return min(1.0, total_latency / total_cycles)


def atom_vs_red_ratio(atomic_df) -> dict:
    """Fraction of atom vs red events."""
    if atomic_df is None or atomic_df.empty:
        return {"atom": 0.0, "red": 0.0}
    n = len(atomic_df)
    atom_count = int((atomic_df["kind"] == "ATOM").sum())
    red_count = int((atomic_df["kind"] == "RED").sum())
    return {"atom": atom_count / n, "red": red_count / n}


def cooperative_epilogue_overlap(bulk_store_df, mma_df) -> float:
    """Fraction of in-flight bulk store cycles during which mma events occurred."""
    if bulk_store_df is None or bulk_store_df.empty:
        return 0.0
    issues = bulk_store_df[bulk_store_df["kind"] == "ISSUE"] if "kind" in bulk_store_df.columns else bulk_store_df
    if issues.empty:
        return 0.0
    total_inflight = 0
    overlapped = 0
    for _, row in issues.iterrows():
        start = int(row["cycle"])
        end = int(row.get("completion_at", start))
        total_inflight += max(0, end - start)
        if mma_df is not None and not mma_df.empty:
            count = int(((mma_df["cycle"] >= start) & (mma_df["cycle"] <= end)).sum())
            if count > 0:
                overlapped += min(end - start, count * 8)
    return overlapped / max(total_inflight, 1)


def bulk_store_async_overlap_ratio(bulk_store_df, warp_state_df) -> float:
    if bulk_store_df is None or bulk_store_df.empty:
        return 0.0
    issues = bulk_store_df[bulk_store_df["kind"] == "ISSUE"]
    if issues.empty:
        return 0.0
    total_inflight = 0
    overlapped = 0
    for _, row in issues.iterrows():
        start = int(row["cycle"])
        end = int(row["completion_at"])
        total_inflight += max(0, end - start)
        if warp_state_df is not None and not warp_state_df.empty:
            for _, ws in warp_state_df.iterrows():
                ws_start = max(start, int(ws["start"]))
                ws_end = min(end, int(ws["end"]))
                if ws_end > ws_start and ws.get("state") not in (
                        "BULK_STORE_WAIT", "IDLE"):
                    overlapped += ws_end - ws_start
    return overlapped / max(total_inflight, 1)


def stream_concurrency_factor(kernel_launch_df, total_cycles: int) -> float:
    """Average number of streams active per cycle, over the device run.
    1.0 = serial; up to N for full overlap."""
    if kernel_launch_df is None or kernel_launch_df.empty or total_cycles <= 0:
        return 0.0
    # For each launch, compute its in-flight cycles; sum and divide
    total_active_cycles = 0
    for _, row in kernel_launch_df.iterrows():
        total_active_cycles += max(0, row["complete_cycle"] - row["launch_cycle"])
    return total_active_cycles / total_cycles


def compute_memory_overlap(events_dfs: dict) -> float:
    """Fraction of compute-event cycles that overlap with memory-event cycles
    on different streams."""
    mma_df = events_dfs.get("mma")
    mem_df = events_dfs.get("memory")
    if mma_df is None or mma_df.empty or mem_df is None or mem_df.empty:
        return 0.0
    overlap = 0
    total = len(mma_df)
    for _, mrow in mma_df.iterrows():
        cycle = mrow["cycle"]
        cross_stream_mem = mem_df[(mem_df["cycle"] == cycle)
                                    & (mem_df["stream_id"] != mrow["stream_id"])]
        if not cross_stream_mem.empty:
            overlap += 1
    return overlap / max(total, 1)


def l2_bandwidth_per_stream(memory_events_df) -> dict:
    """Fraction of L2 requests originating from each stream."""
    if memory_events_df is None or memory_events_df.empty:
        return {}
    counts = memory_events_df.groupby("stream_id").size()
    total = counts.sum()
    return {int(sid): float(cnt) / total for sid, cnt in counts.items()}


def stream_fairness_jain(cta_dispatch_df) -> float:
    """Jain's fairness index over per-stream CTA dispatch counts:
    (Σ x_i)² / (n · Σ x_i²)   where x_i = CTAs dispatched for stream i.
    1.0 = perfectly fair; 1/n = worst case."""
    if cta_dispatch_df is None or cta_dispatch_df.empty:
        return 0.0
    counts = cta_dispatch_df.groupby("stream_id").size().values
    n = len(counts)
    if n == 0: return 0.0
    if n == 1: return 1.0
    sum_x = float(counts.sum())
    sum_x_sq = float((counts ** 2).sum())
    return (sum_x ** 2) / (n * sum_x_sq)
