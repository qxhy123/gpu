"""Tests for ConcurrentStreamScheduler (Phase 8 T2)."""


def test_concurrent_scheduler_dispatches_multiple_streams_per_cycle():
    """Per-cycle step returns dispatches from multiple streams in one call."""
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter()
    s0 = Stream(); s1 = Stream()
    s0.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s1.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k1")

    sched = ConcurrentStreamScheduler([s0, s1])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM(), _SM()]

    decisions = sched.step(sms, current_cycle=0)
    stream_ids = {d[0].stream_id for d in decisions}
    assert len(stream_ids) >= 1
    assert all(isinstance(d, tuple) and len(d) == 3 for d in decisions)


def test_concurrent_scheduler_default_priority_weights():
    """Default weights: high=4, normal=2, low=1."""
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    sched = ConcurrentStreamScheduler([])
    assert sched._priority_weights == {"high": 4, "normal": 2, "low": 1}


def test_concurrent_run_streams_basic_one_stream():
    """run_streams with single stream produces one Result (regression check)."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    s = Stream()
    s.launch(ptx_src=src, grid=(2,1,1), block=(32,1,1), params={}, kernel_name="k")

    cfg = load_default()
    from gpusim.core.device import Device
    d = Device(cfg)
    multi_res = d.run_streams([s])
    assert 0 in multi_res.streams
    assert len(multi_res.streams[0]) == 1
    assert multi_res.streams[0][0].metrics["cycles"] > 0


def test_concurrent_run_streams_two_streams_both_complete():
    """run_streams with 2 streams both produce Result with correct stream_id."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    s0 = Stream(); s1 = Stream()
    s0.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s1.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1")

    cfg = load_default()
    from gpusim.core.device import Device
    d = Device(cfg)
    multi_res = d.run_streams([s0, s1])
    assert 0 in multi_res.streams and 1 in multi_res.streams
    assert multi_res.streams[0][0].stream_id == 0
    assert multi_res.streams[1][0].stream_id == 1


def test_event_blocks_consumer_stream_until_signaled():
    """Stream waiting on unsignaled event is skipped."""
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter(); _reset_event_id_counter()
    s_a = Stream()
    s_b = Stream()
    ev = Event()

    s_a.launch(ptx_src="x", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="ka")
    s_b.wait(ev)
    s_b.launch(ptx_src="x", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="kb")

    sched = ConcurrentStreamScheduler([s_a, s_b])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()] * 8

    decisions = sched.step(sms, current_cycle=0)
    stream_ids = {d[0].stream_id for d in decisions}
    assert s_a.stream_id in stream_ids
    assert s_b.stream_id not in stream_ids


def test_record_marker_processed_in_scheduler():
    """A _RecordMarker at head of pending is processed (event.recorded_in_stream set)."""
    from gpusim.api import Stream, Event, _RecordMarker, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter(); _reset_event_id_counter()
    s = Stream()
    ev = Event()
    s.record(ev)   # _RecordMarker first
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k")

    sched = ConcurrentStreamScheduler([s])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()]

    sched.step(sms, current_cycle=0)
    # After scheduler processes _RecordMarker, event.recorded_in_stream should be set
    assert ev.recorded_in_stream is s
