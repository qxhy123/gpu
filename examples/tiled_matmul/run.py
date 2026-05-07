# examples/tiled_matmul/run.py
import numpy as np, pathlib, gpusim
def main():
    rng = np.random.RandomState(1)
    A = rng.randn(16,16).astype(np.float32); B = rng.randn(16,16).astype(np.float32)
    C = np.zeros((16,16), dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(16,16,1),
               params={"A":A,"B":B,"C":C}, mode="timing")
    print("max abs error:", float(np.max(np.abs(C - A @ B))))
if __name__ == "__main__": main()
