import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    s = Stream()
    s.begin_capture()
    for _ in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT}, kernel_name="inc", config=cfg)
    g = s.end_capture()

    print(f"Captured: {len(g.nodes)} nodes, {len(g.edges)} edges, is_captured={g.is_captured}")

    exec = g.instantiate(cfg)
    for _ in range(5):
        exec.launch()
    print(f"After 5 replays * 3 kernels * 1 inc per thread: OUT.sum() = {OUT.sum()} (expected 480)")


if __name__ == "__main__":
    main()
