import numpy as np
import pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    buf = np.full(n * 4, 99, dtype=np.uint8)
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    g = Graph()
    n0 = g.add_memset_node(buf=buf, value=0, n_bytes=n * 4)
    n1 = g.add_kernel_node(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                           params={"A": A, "B": B, "OUT": OUT},
                           kernel_name="vec_add")
    g.add_dependency(n0, n1)
    n2 = g.add_memset_node(buf=buf, value=0, n_bytes=n * 4)
    g.add_dependency(n1, n2)
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    print(f"Graph (memset+kernel+memset): {cycles} cycles")
    print(f"OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
