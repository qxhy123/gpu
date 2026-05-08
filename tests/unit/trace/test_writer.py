from gpusim.trace.recorder import Recorder
from gpusim.trace.writer import write_parquet
import pyarrow.parquet as pq


def test_write_parquet_creates_three_tables(tmp_path):
    r = Recorder()
    r.warp_state(cycle=0, warp_id=0, state="ISSUED", pc=0)
    r.warp_state(cycle=1, warp_id=0, state="ISSUED", pc=1)
    r.warp_state(cycle=2, warp_id=0, state="SCOREBOARD", pc=2)
    r.instr_issue(cycle=0, warp_id=0, pc=0, op="add.f32", src_loc=("k", 1), active_mask=0xFFFFFFFF)
    r.smem_access(cycle=1, warp_id=0, conflict_degree=2, addresses=[0]*32)
    r.gmem_access(cycle=2, warp_id=0, n_transactions=1, efficiency=1.0, addresses=[i*4 for i in range(32)])
    r.cta_launch(cycle=0, cta_id=0, warps=1, regs=16, smem_bytes=128)
    r.cta_retire(cycle=10, cta_id=0)
    r.div_push(cycle=5, warp_id=0, pc=3, rpc=10, taken_mask=0xFF)

    out = tmp_path / "trace.parquet"
    write_parquet(r, out)
    # parquet writer creates a directory with multiple files
    assert (out / "warp_state.parquet").exists()
    assert (out / "instr_issue.parquet").exists()
    assert (out / "smem.parquet").exists()
    assert (out / "gmem.parquet").exists()
    assert (out / "cta.parquet").exists()
    assert (out / "div.parquet").exists()

    df = pq.read_table(out / "warp_state.parquet").to_pandas()
    # pc=0 ISSUED and pc=1 ISSUED are different PCs → separate segments; plus SCOREBOARD = 3
    assert len(df) == 3  # three segments (different-PC ISSUED entries are not merged)


def test_write_parquet_creates_phase2_tables(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    import pyarrow.parquet as pq
    r = Recorder()
    r.l1_access(cycle=10, warp_id=0, kind="HIT",
                line_addr=0x100, set_idx=0, way=0)
    r.l2_access(cycle=15, kind="HIT", line_addr=0x100, set_idx=0, way=0)
    r.hbm_access(cycle=30, served_at=160, addr=0x100, channel=2, bank=5, row=42,
                 kind="READ", row_kind="ROW_MISS", queue_wait=5)
    out = tmp_path / "trace"
    write_parquet(r, out)
    assert (out / "l1.parquet").exists()
    assert (out / "l2.parquet").exists()
    assert (out / "hbm.parquet").exists()
    df = pq.read_table(out / "l1.parquet").to_pandas()
    assert len(df) == 1
    df = pq.read_table(out / "hbm.parquet").to_pandas()
    assert df.iloc[0]["channel"] == 2
