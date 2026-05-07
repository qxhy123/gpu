# tests/parity/test_vector_add_timing.py
import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/vector_add/kernel.ptx").read_text()

def test_vector_add_timing_mode_correct():
    n = 1024
    rng = np.random.RandomState(7)
    a = rng.randn(n).astype(np.float32); b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    res = gpusim.run(ptx_src=PTX, grid=(8,1,1), block=(128,1,1),
                     params={"A": a, "B": b, "C": c, "N": n}, mode="timing")
    np.testing.assert_allclose(c, a + b, rtol=1e-5)
    assert res.metrics["cycles"] > 0
