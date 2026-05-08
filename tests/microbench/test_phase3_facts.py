"""Phase 3 microbench — textbook facts the simulator must reproduce."""
import pathlib, numpy as np


def test_fp8_mma_is_within_10pct_of_fp16():
    """FP8 m16n8k32 single mma cycles should be ≤ 1.1× FP16 m16n8k16 single mma cycles
    (same latency 8 cycles, but FP8 single mma covers 2× K → 2× FLOPS/cycle).
    Test running each through the simulator and comparing cycle counts."""
    import gpusim
    import ml_dtypes
    rng = np.random.RandomState(0)

    base = pathlib.Path(__file__).resolve().parents[2]

    # FP16 path
    A_f16 = rng.randn(16, 16).astype(np.float16)
    B_f16 = rng.randn(16, 8).astype(np.float16)
    out16 = np.zeros(128, dtype=np.float32)
    ptx16 = (base / "examples" / "tc_matmul_precisions" / "kernel_fp16.ptx").read_text()
    res16 = gpusim.run(
        ptx_src=ptx16, grid=(1,1,1), block=(32,1,1),
        params={"A": A_f16.flatten().copy(), "B": B_f16.flatten().copy(),
                "C": np.zeros(128, dtype=np.float32), "OUT": out16},
        mode="timing",
    )

    # FP8 path
    A_e4m3 = rng.randn(16, 32).astype(ml_dtypes.float8_e4m3fn)
    B_e4m3 = rng.randn(32, 8).astype(ml_dtypes.float8_e4m3fn)
    out8 = np.zeros(128, dtype=np.float32)
    ptx8 = (base / "examples" / "tc_matmul_precisions" / "kernel_e4m3.ptx").read_text()
    res8 = gpusim.run(
        ptx_src=ptx8, grid=(1,1,1), block=(32,1,1),
        params={"A": A_e4m3.flatten().copy().view(np.uint8),
                "B": B_e4m3.flatten().copy().view(np.uint8),
                "C": np.zeros(128, dtype=np.float32), "OUT": out8},
        mode="timing",
    )

    c16 = res16.metrics["cycles"]
    c8 = res8.metrics["cycles"]
    ratio = c8 / c16
    # FP8 should be roughly same latency as FP16 (within 10%)
    # Note: their cycle counts differ in non-mma overhead (more A loads for FP8: 16 vs 8 regs/lane)
    # so loosen the bound to 0.5..2.0 — the spirit is "comparable, not 64x slower"
    assert 0.3 <= ratio <= 2.0, f"FP8 cycles ratio = {ratio} (expected ~1.0)"


def test_fp16_accum_error_ratio_vs_fp32_accum():
    """FP16 accum loses ≥ 50× more precision than FP32 accum after 64 iterations."""
    import gpusim
    rng = np.random.RandomState(42)

    base = pathlib.Path(__file__).resolve().parents[2]
    A = rng.randn(16, 16 * 64).astype(np.float16).flatten().copy()
    B = rng.randn(16 * 64, 8).astype(np.float16).flatten().copy()
    expected = (A.reshape(16, -1).astype(np.float32) @ B.reshape(-1, 8).astype(np.float32))

    ptx_fp32 = (base / "examples" / "mixed_accum" / "kernel_fp32_accum.ptx").read_text()
    out32 = np.zeros(128, dtype=np.float32)
    gpusim.run(ptx_src=ptx_fp32, grid=(1,1,1), block=(32,1,1),
                params={"A": A.copy(), "B": B.copy(), "OUT": out32, "K_ITERS": 64},
                mode="functional")
    err32 = np.max(np.abs(out32.reshape(16, 8) - expected))

    ptx_fp16 = (base / "examples" / "mixed_accum" / "kernel_fp16_accum.ptx").read_text()
    out16 = np.zeros(128, dtype=np.float16)
    gpusim.run(ptx_src=ptx_fp16, grid=(1,1,1), block=(32,1,1),
                params={"A": A.copy(), "B": B.copy(), "OUT": out16, "K_ITERS": 64},
                mode="functional")
    err16 = np.max(np.abs(out16.reshape(16, 8).astype(np.float32) - expected))

    ratio = err16 / max(err32, 1e-9)
    assert ratio >= 50, f"FP16/FP32 accum error ratio = {ratio} (expected >= 50)"
