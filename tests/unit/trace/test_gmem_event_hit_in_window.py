"""Tests for GmemEvent hit + in_window fields (Phase 9 T6)."""


def test_gmem_event_hit_in_window_default():
    from gpusim.trace.events import GmemEvent
    e = GmemEvent(cycle=0, warp_id=0, n_transactions=1, efficiency=1.0,
                  addresses=(0x1000,))
    assert e.hit is False
    assert e.in_window is False


def test_gmem_event_hit_in_window_set():
    from gpusim.trace.events import GmemEvent
    e = GmemEvent(cycle=0, warp_id=0, n_transactions=1, efficiency=1.0,
                  addresses=(0x1000,), hit=True, in_window=True)
    assert e.hit is True
    assert e.in_window is True


def test_recorder_gmem_access_accepts_hit_in_window():
    from gpusim.trace.recorder import Recorder
    r = Recorder()
    r.gmem_access(cycle=0, warp_id=0, n_transactions=1, efficiency=1.0,
                  addresses=[0x1000], hit=True, in_window=True)
    ev = r.gmem_accesses()[-1]
    assert ev.hit is True
    assert ev.in_window is True
