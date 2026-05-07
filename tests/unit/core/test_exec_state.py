import numpy as np
from gpusim.core.exec import ThreadState, WarpFnState, RegName

def test_thread_reg_read_write():
    t = ThreadState()
    t.set_u32("r1", 42)
    assert t.get_u32("r1") == 42

def test_thread_predicate_default_false():
    t = ThreadState()
    assert t.get_pred("p1") is False
    t.set_pred("p1", True)
    assert t.get_pred("p1") is True

def test_warp_active_mask_default_all_active():
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))
    assert w.active_mask == (1 << 32) - 1

def test_per_lane_register():
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))
    for lane in range(32):
        w.threads[lane].set_u32("r1", lane * 10)
    assert [w.threads[i].get_u32("r1") for i in range(32)] == [i*10 for i in range(32)]
