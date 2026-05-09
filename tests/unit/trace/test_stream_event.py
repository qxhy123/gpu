def test_recorder_records_stream_event():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.stream_event(cycle=10, event_id=1, stream_id=0, op="record")
    assert len(r.stream_event_events) == 1
    e = r.stream_event_events[0]
    assert e.op == "record"
    assert e.event_id == 1


def test_recorder_writes_stream_event_parquet(tmp_path):
    from gpusim.trace.recorder import Recorder
    from gpusim.trace.writer import write_parquet
    r = Recorder()
    r.stream_event(cycle=0, event_id=0, stream_id=0, op="record")
    write_parquet(r, tmp_path)
    assert (tmp_path / "stream_event.parquet").exists()
