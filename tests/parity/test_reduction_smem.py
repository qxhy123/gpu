# tests/parity/test_reduction_smem.py
import numpy as np, pathlib, gpusim

PTX = (pathlib.Path(__file__).parents[2] / "examples/reduction_smem/kernel.ptx").read_text()


def test_reduction_smem_correct():
    rng = np.random.RandomState(0)
    a = rng.randint(-100, 100, size=32).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"A": a, "OUT": out}, mode="functional")
    assert int(out[0]) == int(a.sum())
