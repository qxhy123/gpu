import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A1 = np.full(n, 1.0, dtype=np.float32)
    B1 = np.full(n, 1.0, dtype=np.float32)
    A2 = np.full(n, 5.0, dtype=np.float32)
    B2 = np.full(n, 3.0, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A1, "B": B1, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    exec.launch()
    print(f"Replay 1 (A=1, B=1): OUT[0:4] = {list(OUT[0:4])}")
    exec.update_kernel_node_params(0, params={"A": A2, "B": B2, "OUT": OUT})
    exec.launch()
    print(f"Replay 2 (A=5, B=3): OUT[0:4] = {list(OUT[0:4])}")
    print(f"Update count: {exec._update_count}")


if __name__ == "__main__":
    main()
