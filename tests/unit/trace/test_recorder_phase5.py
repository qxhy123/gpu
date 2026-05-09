def test_recorder_records_cluster_dispatch():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.cluster_dispatch(cycle=10, cluster_id=0, cluster_size=4,
                         sm_ids=(0, 1, 2, 3), cta_ids=(0, 1, 2, 3),
                         queue_position=0)
    assert len(r.cluster_dispatch_events) == 1
    e = r.cluster_dispatch_events[0]
    assert e.cluster_id == 0 and e.cluster_size == 4


def test_recorder_records_cluster_barrier():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.cluster_barrier(kind="ARRIVE", cycle=20, cluster_id=0,
                        cta_id=2, rank=2, sm_id=2, arrived_count=1)
    assert len(r.cluster_barrier_events) == 1


def test_writer_phase5_parquets(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.cluster_dispatch(cycle=0, cluster_id=0, cluster_size=2,
                         sm_ids=(0,1), cta_ids=(0,1), queue_position=0)
    r.cluster_barrier(kind="ARRIVE", cycle=1, cluster_id=0, cta_id=0,
                        rank=0, sm_id=0)
    write_parquet(r, tmp_path)
    assert (tmp_path / "cluster_dispatch.parquet").exists()
    assert (tmp_path / "cluster_barrier.parquet").exists()
