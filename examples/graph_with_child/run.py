import numpy as np, pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    inner = Graph()
    n0 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_0")
    n1 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_1")
    inner.add_dependency(n0, n1)
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    exec = outer.instantiate(cfg)
    cycles = exec.launch()
    print(f"Graph (outer with 1 child of 2 kernels): {cycles} cycles")
    print(f"OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
