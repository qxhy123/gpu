# examples/coalescing_demo/run.py
import numpy as np, pathlib, gpusim
def main():
    n = 1024
    a = np.arange(n, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    for stride in [1, 2, 4, 8]:
        out = np.zeros(32, dtype=np.uint32)
        res = gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                         params={"A": a, "OUT": out, "STRIDE": stride}, mode="timing")
        print(f"stride={stride}: cycles={res.metrics['cycles']}")
if __name__ == "__main__": main()
