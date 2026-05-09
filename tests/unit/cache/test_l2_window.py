"""Tests for L2Line.owner_stream_id / in_window (T21) and
L2Cache.register_stream_window (T22)."""


def test_l2_line_default_owner_unset():
    from gpusim.core.cache.l2 import L2Line
    line = L2Line(addr=0x1000, valid=False, dirty=False, last_use=0)
    assert line.owner_stream_id == -1
    assert line.in_window is False


def test_l2_line_owner_settable():
    from gpusim.core.cache.l2 import L2Line
    line = L2Line(addr=0x1000, valid=True, dirty=False, last_use=0,
                  owner_stream_id=2, in_window=True)
    assert line.owner_stream_id == 2
    assert line.in_window is True


def test_l2_register_stream_window():
    from gpusim.core.cache.l2 import L2Cache
    from gpusim.config.schema import CacheConfig

    class _NoOpHbm:
        def request(self, line_addr, now): return now + 100
        def write_request(self, line_addr, now): return now + 100

    cfg = CacheConfig()
    l2 = L2Cache(cfg, _NoOpHbm())
    l2.register_stream_window(stream_id=0, start_set=0, n_sets=32)
    assert l2._stream_windows[0] == (0, 32)
