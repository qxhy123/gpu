"""Tests for Stream.priority field (T9) and SchedulerConfig.priority_weights wiring (T10)."""
import pytest


def test_stream_priority_default_normal():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    assert s.priority == "normal"


def test_stream_priority_high():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream(priority="high")
    assert s.priority == "high"


def test_stream_priority_low():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream(priority="low")
    assert s.priority == "low"


def test_stream_priority_invalid_raises():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    with pytest.raises(ValueError, match="priority must be"):
        Stream(priority="urgent")


def test_scheduler_uses_priority_weights():
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.core.scheduler import ConcurrentStreamScheduler
    _reset_stream_id_counter()
    s_high = Stream(priority="high")
    s_low = Stream(priority="low")
    s_high.launch(ptx_src="x", grid=(8,1,1), block=(32,1,1), params={}, kernel_name="kh")
    s_low.launch(ptx_src="x", grid=(8,1,1), block=(32,1,1), params={}, kernel_name="kl")

    sched = ConcurrentStreamScheduler([s_high, s_low])
    class _SM:
        def __init__(self): self.cap = 100
    sms = [_SM()] * 16

    decisions = sched.step(sms, current_cycle=0)
    counts = {0: 0, 1: 0}
    for s, c, sm in decisions:
        counts[s.stream_id] += 1
    # high gets weight 4, low gets weight 1
    assert counts[s_high.stream_id] == 4
    assert counts[s_low.stream_id] == 1
