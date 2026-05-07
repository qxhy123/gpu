# tests/parity/test_bank_conflict_demo.py
import numpy as np, pathlib, gpusim
PTX = (pathlib.Path(__file__).parents[2]/"examples/bank_conflict_demo/kernel.ptx").read_text()
def test_bank_conflict_demo():
    out = np.zeros(32, dtype=np.uint32)
    gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
               params={"OUT": out}, mode="functional")
    # each lane reads its own value back
    np.testing.assert_array_equal(out, np.arange(32, dtype=np.uint32))
