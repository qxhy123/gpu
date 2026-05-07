from gpusim.core.gmem import coalescing_info

def test_perfectly_coalesced_one_transaction():
    addrs = [i * 4 for i in range(32)]
    info = coalescing_info(addrs, active_mask=(1<<32)-1, sector_bytes=128)
    assert info.n_transactions == 1
    assert info.efficiency == 1.0

def test_stride_2_half_efficiency():
    addrs = [i * 8 for i in range(32)]
    info = coalescing_info(addrs)
    assert info.n_transactions == 2
    assert abs(info.efficiency - 0.5) < 1e-9

def test_stride_4_quarter_efficiency():
    addrs = [i * 16 for i in range(32)]
    info = coalescing_info(addrs)
    assert info.n_transactions == 4
    assert abs(info.efficiency - 0.25) < 1e-9

def test_random_pattern():
    import random
    random.seed(0)
    addrs = [random.randrange(0, 4096) & ~3 for _ in range(32)]
    info = coalescing_info(addrs)
    assert info.n_transactions >= 1
    assert 0.0 < info.efficiency <= 1.0

def test_inactive_lanes_excluded():
    addrs = [0]*32
    mask = 0x0000FFFF
    info = coalescing_info(addrs, active_mask=mask)
    assert info.n_transactions == 1
    assert info.efficiency == 16 / 32
