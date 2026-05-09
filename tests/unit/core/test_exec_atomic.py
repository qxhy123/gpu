def test_smem_atomic_op_add_int():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 16, 100)
    old = s.atomic_op(0, 16, "add", 5, PtxType.u32)
    assert old == 100
    assert s.load_u32(0, 16) == 105


def test_smem_atomic_op_min_max():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 0, 50)
    old = s.atomic_op(0, 0, "min", 30, PtxType.u32)
    assert old == 50
    assert s.load_u32(0, 0) == 30
    old = s.atomic_op(0, 0, "max", 100, PtxType.u32)
    assert old == 30
    assert s.load_u32(0, 0) == 100


def test_smem_atomic_op_cas():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 0, 7)
    old = s.atomic_op(0, 0, "cas", (7, 99), PtxType.u32)
    assert old == 7
    assert s.load_u32(0, 0) == 99
    old = s.atomic_op(0, 0, "cas", (7, 12), PtxType.u32)
    assert old == 99
    assert s.load_u32(0, 0) == 99


def test_smem_atomic_op_exch():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_u32(0, 0, 42)
    old = s.atomic_op(0, 0, "exch", 100, PtxType.u32)
    assert old == 42
    assert s.load_u32(0, 0) == 100


def test_smem_atomic_op_f32():
    from gpusim.core.exec import SharedMemory
    from gpusim.frontend.ir import PtxType
    s = SharedMemory(size_bytes=1024)
    s.allocate_cta(0, 1024)
    s.store_f32(0, 0, 1.5)
    old = s.atomic_op(0, 0, "add", 2.5, PtxType.f32)
    assert old == 1.5
    assert s.load_f32(0, 0) == 4.0


def test_gmem_atomic_op_add_int():
    import numpy as np
    from gpusim.core.exec import GlobalMemory
    from gpusim.frontend.ir import PtxType
    g = GlobalMemory()
    arr = np.array([100], dtype=np.uint32)
    base = g.bind("X", arr)
    old = g.atomic_op(base, "add", 5, PtxType.u32)
    assert old == 100
    assert int(arr[0]) == 105


def test_gmem_atomic_op_cas_match():
    import numpy as np
    from gpusim.core.exec import GlobalMemory
    from gpusim.frontend.ir import PtxType
    g = GlobalMemory()
    arr = np.array([7], dtype=np.uint32)
    base = g.bind("X", arr)
    old = g.atomic_op(base, "cas", (7, 99), PtxType.u32)
    assert old == 7
    assert int(arr[0]) == 99


def test_gmem_atomic_op_cas_mismatch():
    import numpy as np
    from gpusim.core.exec import GlobalMemory
    from gpusim.frontend.ir import PtxType
    g = GlobalMemory()
    arr = np.array([7], dtype=np.uint32)
    base = g.bind("X", arr)
    old = g.atomic_op(base, "cas", (5, 99), PtxType.u32)
    assert old == 7
    assert int(arr[0]) == 7   # unchanged
