import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "tma_store_matmul"


def test_tma_store_matmul_correctness():
    import gpusim
    rng = np.random.RandomState(0)
    A = rng.randn(64, 16).astype(np.float16)
    B = rng.randn(16, 128).astype(np.float16)
    out = np.zeros(64 * 128, dtype=np.float32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(1, 1, 1), block=(128, 1, 1),
        params={"A": A.flatten().copy(), "B": B.flatten().copy(), "OUT": out},
        mode="functional",
    )
    expected = (A.astype(np.float32) @ B.astype(np.float32))
    assert np.allclose(out.reshape(64, 128), expected, atol=2e-2), \
        f"max diff = {np.max(np.abs(out.reshape(64, 128) - expected))}"
