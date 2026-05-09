def test_event_construction_assigns_unique_id():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    e1 = Event()
    e2 = Event()
    assert e1.event_id != e2.event_id
    assert e2.event_id == e1.event_id + 1


def test_event_unsignaled_initially():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    e = Event()
    assert e.recorded_in_stream is None
    assert e.record_cycle is None
    assert e.signaled_at_cycle is None
    assert not e.is_signaled(current_cycle=100)


def test_event_is_signaled_after_signal():
    from gpusim.api import Event, _reset_event_id_counter
    _reset_event_id_counter()
    e = Event()
    e.signaled_at_cycle = 50
    assert not e.is_signaled(current_cycle=49)
    assert e.is_signaled(current_cycle=50)
    assert e.is_signaled(current_cycle=100)
