import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 3
    E = np.arange(n, dtype=np.float32) * 4
    F = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add_a", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F}, kernel_name="vec_add_b", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    print(f"Stream 0 cycles: {multi_res.streams[0][0].metrics['cycles']}")
    print(f"Stream 1 cycles: {multi_res.streams[1][0].metrics['cycles']}")
    print(f"C[0:4] = {list(C[0:4])}")
    print(f"F[0:4] = {list(F[0:4])}")


if __name__ == "__main__":
    main()
