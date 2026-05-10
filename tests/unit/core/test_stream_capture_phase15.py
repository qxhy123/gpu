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
