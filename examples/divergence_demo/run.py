# examples/divergence_demo/run.py
import numpy as np, pathlib, gpusim
def main():
    out = np.zeros(32, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    gpusim.run(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": out}, mode="timing")
    print("output:", out)
if __name__ == "__main__": main()
