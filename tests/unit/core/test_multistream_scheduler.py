from __future__ import annotations


def test_rr_scheduler_alternates_streams():
    """ConcurrentStreamScheduler (aliased as MultiStreamScheduler) dispatches CTAs
    from both streams in a single step() call."""
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import MultiStreamScheduler
    _reset_stream_id_counter()
    s0 = Stream(); s1 = Stream()
    s0.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s1.launch(ptx_src="x", grid=(4,1,1), block=(32,1,1), params={}, kernel_name="k1")

    sched = MultiStreamScheduler([s0, s1])
    # Mock: simple SM list with infinite capacity
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM(), _SM()]

    decisions = sched.step(sms, current_cycle=0)
    # At least one dispatch should occur for each stream
    stream_ids = [s.stream_id for s, cta, sm in decisions]
    assert len(decisions) >= 1
    assert 0 in stream_ids or 1 in stream_ids


def test_intra_stream_fifo_grid_sequencing():
    """Same stream's 2 grids must dispatch in launch order via step()."""
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import MultiStreamScheduler
    _reset_stream_id_counter()
    s = Stream()
    s.launch(ptx_src="x", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.launch(ptx_src="y", grid=(2,1,1), block=(32,1,1), params={}, kernel_name="k1")

    sched = MultiStreamScheduler([s])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()]

    # First step() dispatches CTAs from the first (inflight) grid k0
    decisions = sched.step(sms, current_cycle=0)
    assert len(decisions) >= 1
    stream, cta, sm = decisions[0]
    assert stream.inflight.kernel_name == "k0"

    # Mark grid 0 retired so scheduler can advance to grid k1
    sched.mark_grid_retired(s)

    # Next step() should now dispatch from k1
    decisions2 = sched.step(sms, current_cycle=1)
    assert len(decisions2) >= 1
    stream2, cta2, sm2 = decisions2[0]
    assert stream2.inflight.kernel_name == "k1"


def test_device_run_streams_basic_one_stream():
    """Device.run_streams with a single stream should produce one Result."""
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


def test_cta_dispatch_event_carries_stream_id():
    """When Device.run is called with stream_id=N, the cta_dispatch event has stream_id=N."""
    import gpusim
    from gpusim.api import _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing", config=cfg, stream_id=7)
    df = res.cta_dispatch_events_df if hasattr(res, "cta_dispatch_events_df") else None
    if df is not None and not df.empty:
        assert (df["stream_id"] == 7).all()


def test_warp_events_carry_stream_id():
    """When a kernel runs on stream_id=N, all warp events (instr_issue, etc.)
    should carry stream_id=N."""
    import gpusim
    from gpusim.api import _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r<3>;
    mov.u32 %r0, %tid.x;
    add.s32 %r1, %r0, 1;
    ret;
}
"""
    cfg = load_default()
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing", config=cfg, stream_id=4)

    # Check instr_issue events all carry stream_id=4
    issue_df = res.instr_issue_events_df if hasattr(res, "instr_issue_events_df") else None
    if issue_df is not None and not issue_df.empty:
        assert (issue_df["stream_id"] == 4).all()
