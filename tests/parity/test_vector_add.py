import numpy as np, pathlib
import gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/vector_add/kernel.ptx").read_text()

def test_vector_add_1024():
    n = 1024
    rng = np.random.RandomState(42)
    a = rng.randn(n).astype(np.float32)
    b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
               params={"A": a, "B": b, "C": c, "N": n}, mode="functional")
    np.testing.assert_allclose(c, a + b, rtol=1e-5)
