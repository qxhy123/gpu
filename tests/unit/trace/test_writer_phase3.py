def test_writer_emits_phase3_parquets(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_all
    r = Recorder()
    r.mma(cycle=1, warp_id=0, pc=0, precision="f16", shape_m=16, shape_n=8,
          shape_k=16, accum_dtype="f32", flops_count=4096)
    r.wgmma(kind="ISSUE", cycle=2, warp_group_id=0, pc=1)
    r.tma(cycle=3, completion_at=10, smem_dst=0, gmem_base=0,
          dim_x=8, dim_y=8, bytes_total=128, n_cache_lines=1, mbarrier_addr=0)
    r.mbarrier(kind="FLIP", cycle=10, cta_id=0, smem_addr=0)
    write_all(r, tmp_path)
    assert (tmp_path / "mma.parquet").exists()
    assert (tmp_path / "wgmma.parquet").exists()
    assert (tmp_path / "tma.parquet").exists()
    assert (tmp_path / "mbarrier.parquet").exists()
