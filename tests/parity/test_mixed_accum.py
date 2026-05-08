import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mixed_accum"


def _run(variant: str):
    import gpusim
    rng = np.random.RandomState(42)
    A_full = rng.randn(16, 16 * 64).astype(np.float16)
    B_full = rng.randn(16 * 64, 8).astype(np.float16)
    A = A_full.flatten().copy()
    B = B_full.flatten().copy()
    if variant == "fp16_accum":
        out_dtype = np.float16
    else:
        out_dtype = np.float32
    out = np.zeros(16 * 8, dtype=out_dtype)
    ptx = (_DIR / f"kernel_{variant}.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"A": A, "B": B, "OUT": out, "K_ITERS": 64},
               mode="functional")
    expected = (A_full.astype(np.float32) @ B_full.astype(np.float32))
    return out.reshape(16, 8).astype(np.float32), expected


def test_fp16_accum_loses_precision():
    out, expected = _run("fp16_accum")
    diff = np.max(np.abs(out - expected))
    assert diff > 5e-2, f"FP16 accum should lose precision (got max diff {diff})"


def test_fp32_accum_preserves_precision():
    out, expected = _run("fp32_accum")
    diff = np.max(np.abs(out - expected))
    assert diff < 5e-2, f"FP32 accum should be precise (got max diff {diff})"
