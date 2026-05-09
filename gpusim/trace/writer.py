from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from .recorder import Recorder


def write_parquet(rec: Recorder, out: str | Path) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # warp_state segments
    segs = list(rec.all_warp_segments())
    tbl_ws = pa.table({
        "warp_id":  [s.warp_id for s in segs],
        "start":    [s.start for s in segs],
        "end":      [s.end for s in segs],
        "state":    [s.state for s in segs],
        "pc":       [s.pc for s in segs],
    })
    pq.write_table(tbl_ws, out / "warp_state.parquet")

    issues = rec.instr_issues()
    tbl_i = pa.table({
        "cycle": [e.cycle for e in issues],
        "warp_id": [e.warp_id for e in issues],
        "pc": [e.pc for e in issues],
        "op": [e.op for e in issues],
        "file": [e.src_loc[0] for e in issues],
        "line": [e.src_loc[1] for e in issues],
        "active_mask": [e.active_mask for e in issues],
    })
    pq.write_table(tbl_i, out / "instr_issue.parquet")

    s_evs = rec.smem_accesses()
    tbl_s = pa.table({
        "cycle": [e.cycle for e in s_evs],
        "warp_id": [e.warp_id for e in s_evs],
        "conflict_degree": [e.conflict_degree for e in s_evs],
    })
    pq.write_table(tbl_s, out / "smem.parquet")

    g_evs = rec.gmem_accesses()
    tbl_g = pa.table({
        "cycle": [e.cycle for e in g_evs],
        "warp_id": [e.warp_id for e in g_evs],
        "n_transactions": [e.n_transactions for e in g_evs],
        "efficiency": [e.efficiency for e in g_evs],
    })
    pq.write_table(tbl_g, out / "gmem.parquet")

    cta_evs = rec.cta_events()
    tbl_c = pa.table({
        "kind": [e.kind for e in cta_evs],
        "cycle": [e.cycle for e in cta_evs],
        "cta_id": [e.cta_id for e in cta_evs],
        "warps": [e.warps for e in cta_evs],
        "regs": [e.regs for e in cta_evs],
        "smem_bytes": [e.smem_bytes for e in cta_evs],
    })
    pq.write_table(tbl_c, out / "cta.parquet")

    d_evs = rec.div_events()
    tbl_d = pa.table({
        "kind": [e.kind for e in d_evs],
        "cycle": [e.cycle for e in d_evs],
        "warp_id": [e.warp_id for e in d_evs],
        "pc": [e.pc for e in d_evs],
        "rpc": [e.rpc for e in d_evs],
        "taken_mask": [e.taken_mask for e in d_evs],
    })
    pq.write_table(tbl_d, out / "div.parquet")

    # Phase 2: l1 / l2 / hbm
    l1_evs = rec.l1_accesses()
    tbl_l1 = pa.table({
        "cycle":     [e.cycle for e in l1_evs],
        "warp_id":   [e.warp_id for e in l1_evs],
        "kind":      [e.kind for e in l1_evs],
        "line_addr": [e.line_addr for e in l1_evs],
        "set_idx":   [e.set_idx for e in l1_evs],
        "way":       [e.way for e in l1_evs],
        "mshr_slot": [e.mshr_slot if e.mshr_slot is not None else -1
                      for e in l1_evs],
    })
    pq.write_table(tbl_l1, out / "l1.parquet")

    l2_evs = rec.l2_accesses()
    tbl_l2 = pa.table({
        "cycle":       [e.cycle for e in l2_evs],
        "kind":        [e.kind for e in l2_evs],
        "line_addr":   [e.line_addr for e in l2_evs],
        "set_idx":     [e.set_idx for e in l2_evs],
        "way":         [e.way for e in l2_evs],
        "victim_addr": [e.victim_addr for e in l2_evs],
    })
    pq.write_table(tbl_l2, out / "l2.parquet")

    hbm_evs = rec.hbm_accesses()
    tbl_hbm = pa.table({
        "cycle":      [e.cycle for e in hbm_evs],
        "served_at":  [e.served_at for e in hbm_evs],
        "addr":       [e.addr for e in hbm_evs],
        "channel":    [e.channel for e in hbm_evs],
        "bank":       [e.bank for e in hbm_evs],
        "row":        [e.row for e in hbm_evs],
        "kind":       [e.kind for e in hbm_evs],
        "row_kind":   [e.row_kind for e in hbm_evs],
        "queue_wait": [e.queue_wait for e in hbm_evs],
    })
    pq.write_table(tbl_hbm, out / "hbm.parquet")

    # Phase 3: mma / wgmma / tma / mbarrier
    if rec.mma_events:
        tbl_mma = pa.table({
            "cycle":       [e.cycle for e in rec.mma_events],
            "warp_id":     [e.warp_id for e in rec.mma_events],
            "pc":          [e.pc for e in rec.mma_events],
            "precision":   [e.precision for e in rec.mma_events],
            "shape_m":     [e.shape_m for e in rec.mma_events],
            "shape_n":     [e.shape_n for e in rec.mma_events],
            "shape_k":     [e.shape_k for e in rec.mma_events],
            "accum_dtype": [e.accum_dtype for e in rec.mma_events],
            "flops_count": [e.flops_count for e in rec.mma_events],
        })
        pq.write_table(tbl_mma, out / "mma.parquet")

    if rec.wgmma_events:
        tbl_wgmma = pa.table({
            "kind":            [e.kind for e in rec.wgmma_events],
            "cycle":           [e.cycle for e in rec.wgmma_events],
            "warp_group_id":   [e.warp_group_id for e in rec.wgmma_events],
            "pc":              [e.pc for e in rec.wgmma_events],
            "precision":       [e.precision for e in rec.wgmma_events],
            "shape_m":         [e.shape_m for e in rec.wgmma_events],
            "shape_n":         [e.shape_n for e in rec.wgmma_events],
            "shape_k":         [e.shape_k for e in rec.wgmma_events],
            "accum_dtype":     [e.accum_dtype for e in rec.wgmma_events],
            "commit_group_id": [e.commit_group_id for e in rec.wgmma_events],
            "wait_n":          [e.wait_n for e in rec.wgmma_events],
            "completion_at":   [e.completion_at for e in rec.wgmma_events],
        })
        pq.write_table(tbl_wgmma, out / "wgmma.parquet")

    if rec.tma_events:
        tbl_tma = pa.table({
            "cycle":         [e.cycle for e in rec.tma_events],
            "completion_at": [e.completion_at for e in rec.tma_events],
            "smem_dst":      [e.smem_dst for e in rec.tma_events],
            "gmem_base":     [e.gmem_base for e in rec.tma_events],
            "dim_x":         [e.dim_x for e in rec.tma_events],
            "dim_y":         [e.dim_y for e in rec.tma_events],
            "bytes_total":   [e.bytes_total for e in rec.tma_events],
            "n_cache_lines": [e.n_cache_lines for e in rec.tma_events],
            "mbarrier_addr": [e.mbarrier_addr for e in rec.tma_events],
        })
        pq.write_table(tbl_tma, out / "tma.parquet")

    if rec.mbarrier_events:
        tbl_mbarrier = pa.table({
            "kind":        [e.kind for e in rec.mbarrier_events],
            "cycle":       [e.cycle for e in rec.mbarrier_events],
            "cta_id":      [e.cta_id for e in rec.mbarrier_events],
            "smem_addr":   [e.smem_addr for e in rec.mbarrier_events],
            "expected":    [e.expected for e in rec.mbarrier_events],
            "arrived":     [e.arrived for e in rec.mbarrier_events],
            "phase":       [e.phase for e in rec.mbarrier_events],
            "pred_result": [e.pred_result for e in rec.mbarrier_events],
        })
        pq.write_table(tbl_mbarrier, out / "mbarrier.parquet")

    # Phase 4: cta_dispatch / l2_mshr / bulk_store
    if rec.cta_dispatch_events:
        pd.DataFrame([asdict(e) for e in rec.cta_dispatch_events]).to_parquet(
            out / "cta_dispatch.parquet", index=False)
    if rec.l2_mshr_events:
        pd.DataFrame([asdict(e) for e in rec.l2_mshr_events]).to_parquet(
            out / "l2_mshr.parquet", index=False)
    if rec.bulk_store_events:
        pd.DataFrame([asdict(e) for e in rec.bulk_store_events]).to_parquet(
            out / "bulk_store.parquet", index=False)
    if rec.cluster_dispatch_events:
        pd.DataFrame([asdict(e) for e in rec.cluster_dispatch_events]).to_parquet(
            out / "cluster_dispatch.parquet", index=False)
    if rec.cluster_barrier_events:
        pd.DataFrame([asdict(e) for e in rec.cluster_barrier_events]).to_parquet(
            out / "cluster_barrier.parquet", index=False)
    if rec.atomic_events:
        pd.DataFrame([asdict(e) for e in rec.atomic_events]).to_parquet(
            out / "atomic.parquet", index=False)
    if rec.kernel_launch_events:
        pd.DataFrame([asdict(e) for e in rec.kernel_launch_events]).to_parquet(
            out / "kernel_launch.parquet", index=False)


# write_all is the canonical name used by Phase 3 callers; delegates to write_parquet
write_all = write_parquet
