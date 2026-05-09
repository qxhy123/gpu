def test_stream_wait_all_appends_all_events():
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    _reset_stream_id_counter(); _reset_event_id_counter()
    s = Stream()
    ev1 = Event(); ev2 = Event(); ev3 = Event()
    s.wait_all([ev1, ev2, ev3])
    assert ev1 in s.event_waits
    assert ev2 in s.event_waits
    assert ev3 in s.event_waits


def test_stream_wait_all_empty_list_noop():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.wait_all([])
    assert s.event_waits == []
