def test_recorder_records_atomic_event():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.atomic(cycle=10, sm_id=0, warp_id=0, kind="ATOM",
              op="add", space="global", line_addr=0x1000,
              latency=20, n_lanes=4, queue_depth_before=0)
    assert len(r.atomic_events) == 1
    e = r.atomic_events[0]
    assert e.kind == "ATOM" and e.op == "add"


def test_recorder_writes_atomic_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.atomic(cycle=0, sm_id=0, warp_id=0, kind="RED",
              op="add", space="shared", line_addr=64, latency=24)
    write_parquet(r, tmp_path)
    assert (tmp_path / "atomic.parquet").exists()
