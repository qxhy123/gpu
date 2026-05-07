import numpy as np
from gpusim.core.exec import GlobalMemory, SharedMemory, ParamSpace

def test_global_memory_load_store_f32():
    g = GlobalMemory()
    arr = np.arange(16, dtype=np.float32)
    base = g.bind("A", arr)
    assert g.load_f32(base + 4 * 3) == 3.0
    g.store_f32(base + 4 * 5, 99.0)
    assert g.load_f32(base + 4 * 5) == 99.0
    assert arr[5] == 99.0

def test_global_memory_load_u32_round_trip():
    g = GlobalMemory()
    arr = np.zeros(8, dtype=np.uint32)
    base = g.bind("X", arr)
    g.store_u32(base, 0xDEADBEEF)
    assert g.load_u32(base) == 0xDEADBEEF

def test_param_space_returns_value():
    p = ParamSpace({"A": 0xDEAD0000, "N": 1024})
    assert p.read_u64("A") == 0xDEAD0000
    assert p.read_u32("N") == 1024

def test_shared_memory_per_cta_isolated():
    s = SharedMemory(size_bytes=2048)
    s.allocate_cta(cta_id=0, size_bytes=512)
    s.allocate_cta(cta_id=1, size_bytes=512)
    s.store_f32(cta_id=0, offset=0, value=1.0)
    s.store_f32(cta_id=1, offset=0, value=2.0)
    assert s.load_f32(cta_id=0, offset=0) == 1.0
    assert s.load_f32(cta_id=1, offset=0) == 2.0
