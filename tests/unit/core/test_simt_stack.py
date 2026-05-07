from gpusim.core.simt_stack import SIMTStack, SIMTEntry

def test_initial_entry_full_mask():
    s = SIMTStack(warp_size=32, entry_pc=0)
    assert s.top().active_mask == (1 << 32) - 1
    assert s.top().pc == 0

def test_push_diverge_two_paths():
    s = SIMTStack(warp_size=32, entry_pc=0)
    taken = 0xFFFF        # lanes 0..15
    s.diverge(taken_pc=10, fallthrough_pc=5, taken_mask=taken, rpc=20)
    seen_pcs = []
    while s.top().pc != 20:
        seen_pcs.append(s.top().pc)
        e = s.top()
        s.update_top_pc(e.rpc)
        s.maybe_pop()
    assert sorted(seen_pcs) == sorted([5, 10])

def test_no_diverge_when_all_lanes_take_same_path():
    s = SIMTStack(warp_size=32, entry_pc=0)
    full = (1 << 32) - 1
    diverged = s.diverge(taken_pc=10, fallthrough_pc=5, taken_mask=full, rpc=20)
    assert diverged is False
    assert s.top().pc == 10
    assert s.top().active_mask == full
