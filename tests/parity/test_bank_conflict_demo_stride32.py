import numpy as np, pathlib, gpusim

P1 = pathlib.Path(__file__).parents[2] / "examples/bank_conflict_demo/kernel.ptx"
P32 = pathlib.Path(__file__).parents[2] / "examples/bank_conflict_demo/kernel_stride32.ptx"


def test_stride_32_costs_more_cycles():
    out1 = np.zeros(32, dtype=np.uint32)
    res1 = gpusim.run(ptx_src=P1.read_text(), grid=(1,1,1), block=(32,1,1),
                     params={"OUT": out1}, mode="timing")
    out32 = np.zeros(32, dtype=np.uint32)
    res32 = gpusim.run(ptx_src=P32.read_text(), grid=(1,1,1), block=(32,1,1),
                     params={"OUT": out32}, mode="timing")
    assert res32.metrics["cycles"] >= res1.metrics["cycles"] + 25
