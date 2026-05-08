import numpy as np, pathlib, gpusim
def main():
    rng = np.random.RandomState(0)
    A = rng.randn(16,16).astype(np.float32)
    B = rng.randn(16,16).astype(np.float32)
    here = pathlib.Path(__file__).parent
    for variant in ("kernel_smem.ptx", "kernel_no_smem.ptx"):
        C = np.zeros((16,16), dtype=np.float32)
        ptx = (here / variant).read_text()
        res = gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(16,16,1),
                         params={"A":A, "B":B, "C":C}, mode="timing")
        max_err = float(np.max(np.abs(C - A @ B)))
        print(f"{variant}: cycles={res.metrics['cycles']}, max_err={max_err:.6f}")
if __name__ == "__main__": main()
