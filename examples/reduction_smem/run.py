# examples/reduction_smem/run.py
import numpy as np, pathlib, gpusim
def main():
    rng = np.random.RandomState(0)
    a = rng.randint(-100, 100, size=32).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"A": a, "OUT": out}, mode="timing")
    print(f"sum: {out[0]} (expected {a.sum()})")
if __name__ == "__main__": main()
