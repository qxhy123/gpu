import numpy as np, pathlib
import gpusim

def main():
    n = 1024
    rng = np.random.RandomState(42)
    a = rng.randn(n).astype(np.float32)
    b = rng.randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(8,1,1), block=(128,1,1),
               params={"A": a, "B": b, "C": c, "N": n}, mode="functional")
    print("max abs error:", float(np.max(np.abs(c - (a + b)))))

if __name__ == "__main__":
    main()
