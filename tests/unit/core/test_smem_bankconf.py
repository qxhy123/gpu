from gpusim.core.smem import bank_conflict_degree

def test_no_conflict_stride_1():
    addrs = [i * 4 for i in range(32)]
    assert bank_conflict_degree(addrs) == 1

def test_full_conflict_stride_32_words():
    addrs = [i * 4 * 32 for i in range(32)]
    assert bank_conflict_degree(addrs) == 32

def test_broadcast_same_address():
    addrs = [0] * 32
    assert bank_conflict_degree(addrs) == 1

def test_two_way_stride_2_words():
    addrs = [i * 4 * 2 for i in range(32)]
    assert bank_conflict_degree(addrs) == 2

def test_inactive_lanes_ignored():
    addrs = [0] * 32; mask = 0xFF
    assert bank_conflict_degree(addrs, active_mask=mask) == 1

def test_eight_lanes_to_eight_banks_no_conflict():
    addrs = [i * 4 for i in range(8)] + [0] * 24
    mask = 0xFF
    assert bank_conflict_degree(addrs, active_mask=mask) == 1
