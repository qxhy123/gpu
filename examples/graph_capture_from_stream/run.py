import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    for i in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": OUT},
                  kernel_name=f"vec_add_{i}", config=cfg)
    g = s.end_capture()
    print(f"Captured graph: {len(g.nodes)} nodes, {len(g.edges)} edges")
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    print(f"Replay: {cycles} cycles, OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
