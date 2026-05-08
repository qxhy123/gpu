import numpy as np
import pathlib
import gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/bw_saturation_demo/kernel.ptx").read_text()


def test_bw_saturation_demo():
    # 8 CTAs × 32 threads = 256 threads → processes elements [0..255]
    n_threads = 8 * 32
    n = 4096
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    gpusim.run(ptx_src=PTX, grid=(8, 1, 1), block=(32, 1, 1),
               params={"A": a, "OUT": out}, mode="functional")
    np.testing.assert_allclose(out[:n_threads], a[:n_threads], rtol=1e-5)
