import pytest


def test_begin_capture_default_mode_global_works():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()                    # default mode="global"
    g = s.end_capture()
    assert g is not None


def test_begin_capture_explicit_global_works():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture(mode="global")
    g = s.end_capture()
    assert g is not None


def test_begin_capture_unknown_mode_raises():
    from gpusim.api import Stream
    s = Stream()
    with pytest.raises(ValueError, match="only 'global' capture mode supported"):
        s.begin_capture(mode="thread")


def test_begin_capture_double_begin_raises():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    with pytest.raises(RuntimeError, match="already capturing"):
        s.begin_capture()


def test_end_capture_without_begin_raises():
    from gpusim.api import Stream
    s = Stream()
    with pytest.raises(RuntimeError, match="not capturing"):
        s.end_capture()


def test_end_capture_marks_graph_is_captured():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    g = s.end_capture()
    assert g.is_captured is True


def test_recorder_records_stream_capture_begin():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_begin(stream_id=3, cycle=0)
    assert len(rec.stream_capture_begin_events) == 1
    ev = rec.stream_capture_begin_events[0]
    assert ev.stream_id == 3
    assert ev.cycle == 0


def test_recorder_records_stream_capture_end():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_end(stream_id=3, cycle=10, captured_node_count=5)
    assert len(rec.stream_capture_end_events) == 1
    ev = rec.stream_capture_end_events[0]
    assert ev.stream_id == 3
    assert ev.cycle == 10
    assert ev.captured_node_count == 5


def test_stream_begin_capture_with_recorder_emits_begin_event():
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    s = Stream()
    s._recorder = rec
    s.begin_capture()
    assert len(rec.stream_capture_begin_events) == 1
    assert rec.stream_capture_begin_events[0].stream_id == s.stream_id


def test_stream_end_capture_with_recorder_emits_end_event_with_node_count():
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    s = Stream()
    s._recorder = rec
    s.begin_capture()
    s.end_capture()
    assert len(rec.stream_capture_end_events) == 1
    assert rec.stream_capture_end_events[0].captured_node_count == 0
