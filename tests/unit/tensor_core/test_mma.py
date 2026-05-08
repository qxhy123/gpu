import numpy as np
from gpusim.core.tensor_core.mma import execute_mma
from gpusim.core.tensor_core.mma_spec import parse_mma_op
from gpusim.core.exec import WarpFnState
from gpusim.frontend.ir import Reg, RegGroup, PtxType


def _setup_warp_with_matrix(M, K, dtype_np, prefix, w):
    """Distribute an M*K matrix into warp lane registers per the layout in spec §4.1."""
    arr = np.arange(M * K, dtype=np.float32).reshape(M, K)
    arr_typed = arr.astype(dtype_np)
    regs_per_lane = K // 2
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * (K // 2)
        for j in range(regs_per_lane):
            val = float(arr_typed[row, col_base + j])
            w.threads[lane].set_f32(f"{prefix}{j}", val)
    return arr_typed


def test_execute_mma_fp16_k16_matches_numpy():
    """16x8x16 mma (FP16 in/out, FP32 accum) — numerically matches numpy reference."""
    spec = parse_mma_op("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))

    A_ref = _setup_warp_with_matrix(16, 16, np.float16, "a", w)
    B_arr = (np.arange(16*8, dtype=np.float32) * 0.01).reshape(16, 8).astype(np.float16)
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * 4
        for j in range(4):
            w.threads[lane].set_f32(f"b{j}", float(B_arr[row, col_base + j]))
    for lane in range(32):
        for j in range(4):
            w.threads[lane].set_f32(f"c{j}", 0.0)
    dst = RegGroup(regs=tuple(Reg(name=f"d{j}", type=PtxType.f32) for j in range(4)))
    a = RegGroup(regs=tuple(Reg(name=f"a{j}", type=PtxType.f16) for j in range(8)))
    b = RegGroup(regs=tuple(Reg(name=f"b{j}", type=PtxType.f16) for j in range(4)))
    c = RegGroup(regs=tuple(Reg(name=f"c{j}", type=PtxType.f32) for j in range(4)))
    execute_mma(spec, w, dst, a, b, c)
    D = np.zeros((16, 8), dtype=np.float32)
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * 4
        for j in range(4):
            D[row, col_base + j] = w.threads[lane].get_f32(f"d{j}")
    expected = (A_ref.astype(np.float32) @ B_arr.astype(np.float32))
    assert np.allclose(D, expected, atol=1e-2), f"max diff = {np.max(np.abs(D - expected))}"


def test_execute_mma_fp8_k32_matches_numpy_with_loose_tol():
    spec = parse_mma_op("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
    w = WarpFnState(warp_size=32, tids=tuple(range(32)))
    import ml_dtypes
    fp8 = ml_dtypes.float8_e4m3fn
    A = (np.random.RandomState(0).randn(16, 32) * 0.5).astype(fp8)
    B = (np.random.RandomState(1).randn(32, 8) * 0.5).astype(fp8)
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * 16
        for j in range(16):
            w.threads[lane].set_f32(f"a{j}", float(A[row, col_base + j]))
    for lane in range(32):
        for j in range(8):
            w.threads[lane].set_f32(f"b{j}", float(B[lane, j]))
    for lane in range(32):
        for j in range(4):
            w.threads[lane].set_f32(f"c{j}", 0.0)
    dst = RegGroup(regs=tuple(Reg(name=f"d{j}", type=PtxType.f32) for j in range(4)))
    a = RegGroup(regs=tuple(Reg(name=f"a{j}", type=PtxType.e4m3) for j in range(16)))
    b = RegGroup(regs=tuple(Reg(name=f"b{j}", type=PtxType.e4m3) for j in range(8)))
    c = RegGroup(regs=tuple(Reg(name=f"c{j}", type=PtxType.f32) for j in range(4)))
    execute_mma(spec, w, dst, a, b, c)
    D = np.zeros((16, 8), dtype=np.float32)
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * 4
        for j in range(4):
            D[row, col_base + j] = w.threads[lane].get_f32(f"d{j}")
    expected = (A.astype(np.float32) @ B.astype(np.float32))
    assert np.allclose(D, expected, atol=2e-1, rtol=2e-1), f"max diff = {np.max(np.abs(D - expected))}"
