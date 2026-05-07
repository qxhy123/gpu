# tests/parity/test_coalescing_demo.py
import numpy as np, pathlib, gpusim
PTX = (pathlib.Path(__file__).parents[2]/"examples/coalescing_demo/kernel.ptx").read_text()
def test_coalescing_demo():
    n = 1024
    a = np.arange(n, dtype=np.uint32)
    out = np.zeros(32, dtype=np.uint32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"A": a, "OUT": out, "STRIDE": 1}, mode="functional")
    np.testing.assert_array_equal(out, a[:32])
