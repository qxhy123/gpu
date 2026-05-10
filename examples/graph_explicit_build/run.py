import numpy as np
import pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    g = Graph()
    nids = []
    for i in range(3):
        nid = g.add_kernel_node(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                                params={"A": A, "B": B, "OUT": OUT},
                                kernel_name=f"vec_add_{i}")
        nids.append(nid)
    g.add_dependency(nids[0], nids[1])
    g.add_dependency(nids[1], nids[2])
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    print(f"Graph (3-kernel chain): {cycles} cycles")
    print(f"OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
