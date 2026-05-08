import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/l1_thrash_demo/kernel.ptx").read_text()


def test_l1_thrash_demo():
    n = 1024
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(32, dtype=np.float32)
    gpusim.run(
        ptx_src=PTX, grid=(1, 1, 1), block=(32, 1, 1),
        params={"A": a, "OUT": out, "K": 4, "STRIDE": 32, "OUTER_LOOPS": 2},
        mode="functional",
    )
    # last value: a[(K-1)*STRIDE + tid] = a[3*32 + tid] = a[96 + tid]
    expected = a[96 : 96 + 32].copy()
    np.testing.assert_array_equal(out, expected)
