from gpusim.core.scoreboard import Scoreboard

def test_no_initial_dep():
    s = Scoreboard()
    assert s.ready_at("r1", now=0) == 0
    assert s.has_pending("r1", now=0) is False

def test_write_then_read_blocked_until_latency_done():
    s = Scoreboard()
    s.mark_write("r1", available_at_cycle=10)
    assert s.has_pending("r1", now=5) is True
    assert s.ready_at("r1", now=5) == 10
    assert s.has_pending("r1", now=10) is False

def test_multiple_writes_take_max():
    s = Scoreboard()
    s.mark_write("r1", available_at_cycle=10)
    s.mark_write("r1", available_at_cycle=15)
    assert s.ready_at("r1", now=0) == 15

def test_clear_after_completion():
    s = Scoreboard()
    s.mark_write("r1", available_at_cycle=10)
    s.advance(now=12)
    assert s.has_pending("r1", now=12) is False
