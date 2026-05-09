from __future__ import annotations


def test_rr_scheduler_alternates_streams():
    """With 2 streams each having 1 grid of 4 CTAs, RR alternates pick order."""
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

    pick_order = []
    for _ in range(8):
        choice = sched.next_dispatch(sms)
        if choice is None: break
        stream, cta, sm = choice
        pick_order.append(stream.stream_id)

    # RR alternation: 8 picks across 2 streams → each stream gets 4
    counts = {0: pick_order.count(0), 1: pick_order.count(1)}
    assert counts[0] == counts[1] == 4


def test_intra_stream_fifo_grid_sequencing():
    """Same stream's 2 grids must dispatch in launch order."""
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

    grid_order = []
    # Pick 2 CTAs (full grid k0)
    for _ in range(2):
        choice = sched.next_dispatch(sms)
        assert choice is not None
        stream, cta, sm = choice
        grid_order.append(stream.inflight.kernel_name)
    # Mark grid 0 retired so scheduler can advance to grid 1
    sched.mark_grid_retired(s)
    for _ in range(2):
        choice = sched.next_dispatch(sms)
        assert choice is not None
        stream, cta, sm = choice
        grid_order.append(stream.inflight.kernel_name)

    assert grid_order[:2] == ["k0", "k0"]
    assert grid_order[2:] == ["k1", "k1"]
