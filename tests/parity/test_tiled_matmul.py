# tests/parity/test_tiled_matmul.py
import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/tiled_matmul/kernel.ptx").read_text()


def test_tiled_matmul_16x16_correct():
    rng = np.random.RandomState(1)
    A = rng.randn(16, 16).astype(np.float32)
    B = rng.randn(16, 16).astype(np.float32)
    C = np.zeros((16, 16), dtype=np.float32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(16,16,1),
               params={"A": A, "B": B, "C": C}, mode="functional")
    np.testing.assert_allclose(C, A @ B, rtol=1e-4, atol=1e-4)
