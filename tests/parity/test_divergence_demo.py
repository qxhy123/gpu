# tests/parity/test_divergence_demo.py
import numpy as np, pathlib, gpusim
PTX = (pathlib.Path(__file__).parents[2]/"examples/divergence_demo/kernel.ptx").read_text()
def test_divergence_demo():
    out = np.zeros(32, dtype=np.uint32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"OUT": out}, mode="functional")
    expected = np.array([100 if i < 16 else 200 for i in range(32)], dtype=np.uint32)
    np.testing.assert_array_equal(out, expected)
