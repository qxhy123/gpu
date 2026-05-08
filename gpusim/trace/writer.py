from __future__ import annotations
from pathlib import Path
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
