import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "tc_matmul_precisions"


def _run_variant(variant: str, dtype_in_bytes: int, tol: float):
    import gpusim
    from examples.tc_matmul_precisions.reference import build_inputs, reference_output, output_dtype

    A, B, C = build_inputs(variant, seed=0)
    out_dtype = output_dtype(variant)
    out = np.zeros(16 * 8, dtype=out_dtype)

    ptx = (_DIR / f"kernel_{variant}.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
        params={"A": A.flatten().copy(), "B": B.flatten().copy(),
                "C": C.flatten().copy(), "OUT": out},
        mode="functional",
    )
    expected = reference_output(A, B, C, variant)
    out_2d = out.reshape(16, 8)
    assert np.allclose(out_2d.astype(np.float32), expected.astype(np.float32),
                       atol=tol, rtol=tol), \
        f"{variant}: max diff = {np.max(np.abs(out_2d.astype(np.float32) - expected.astype(np.float32)))}"


def test_fp32_baseline():
    _run_variant("fp32", 4, tol=1e-5)
def test_fp16():
    _run_variant("fp16", 2, tol=1e-2)
def test_bf16():
    _run_variant("bf16", 2, tol=1e-2)
def test_e4m3():
    _run_variant("e4m3", 1, tol=2e-1)
def test_tf32():
    _run_variant("tf32", 4, tol=1e-3)
def test_int8():
    _run_variant("int8", 1, tol=0)   # int8 mma is exact
