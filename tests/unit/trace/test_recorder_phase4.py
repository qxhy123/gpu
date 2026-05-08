def test_recorder_records_cta_dispatch():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.cta_dispatch(cycle=10, cta_id=3, sm_id=5,
                    queue_position=2, active_warps_at_dispatch=4)
    assert len(r.cta_dispatch_events) == 1
    e = r.cta_dispatch_events[0]
    assert e.cta_id == 3 and e.sm_id == 5


def test_recorder_records_l2_mshr():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.l2_mshr(kind="ALLOC", cycle=20, line_addr=0x1000, sm_id=2,
                n_waiters=1)
    assert len(r.l2_mshr_events) == 1
    assert r.l2_mshr_events[0].kind == "ALLOC"


def test_recorder_records_bulk_store():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.bulk_store(kind="ISSUE", cycle=30, warp_group_id=0, sm_id=1, pc=5,
                   smem_src=0, gmem_base=0x10000, bytes_total=1024,
                   completion_at=50)
    assert len(r.bulk_store_events) == 1


def test_recorder_writes_phase4_parquets(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.cta_dispatch(cycle=0, cta_id=0, sm_id=0,
                    queue_position=0, active_warps_at_dispatch=0)
    r.l2_mshr(kind="ALLOC", cycle=1, line_addr=0, sm_id=0)
    r.bulk_store(kind="ISSUE", cycle=2, warp_group_id=0, sm_id=0, pc=0,
                   smem_src=0, gmem_base=0, bytes_total=64, completion_at=10)
    write_parquet(r, tmp_path)
    assert (tmp_path / "cta_dispatch.parquet").exists()
    assert (tmp_path / "l2_mshr.parquet").exists()
    assert (tmp_path / "bulk_store.parquet").exists()
