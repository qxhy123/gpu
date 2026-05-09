def test_event_elapsed_time_basic():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    start = Event()
    end = Event()
    start.signaled_at_cycle = 100
    end.signaled_at_cycle = 350
    assert Event.elapsed_time(start, end) == 250


def test_event_elapsed_time_raises_when_unsigned():
    from gpusim.api import Event, _reset_event_id_counter
    import pytest
    _reset_event_id_counter()
    start = Event()
    end = Event()
    end.signaled_at_cycle = 100
    with pytest.raises(RuntimeError, match="not signaled"):
        Event.elapsed_time(start, end)


def test_event_elapsed_time_zero_when_same_cycle():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    start = Event(); end = Event()
    start.signaled_at_cycle = 50
    end.signaled_at_cycle = 50
    assert Event.elapsed_time(start, end) == 0
