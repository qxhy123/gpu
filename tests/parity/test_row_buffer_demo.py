import numpy as np
import pathlib
import gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/row_buffer_demo/kernel.ptx").read_text()


def test_row_buffer_demo():
    n = 1 << 20  # 1 MB array (256K floats)
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(32, dtype=np.float32)
    gpusim.run(ptx_src=PTX, grid=(1, 1, 1), block=(32, 1, 1),
               params={"A": a, "OUT": out, "STRIDE": 1}, mode="functional")
    np.testing.assert_array_equal(out, a[:32])
