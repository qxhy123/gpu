import numpy as np, pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    for probe_value in (10, 2):
        OUT_A = np.zeros(n, dtype=np.uint32)
        OUT_B = np.zeros(n, dtype=np.uint32)
        probe = np.array([probe_value], dtype=np.int32)

        g_A = Graph()
        g_A.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                params={"OUT": OUT_A}, kernel_name="A")
        g_B = Graph()
        g_B.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                params={"OUT": OUT_B}, kernel_name="B")

        g = Graph()
        g.add_conditional_node(cond_fn=lambda p=probe: p[0] > 5,
                                  true_graph=g_A, false_graph=g_B)
        g.instantiate(cfg).launch()
        taken = "A (true)" if probe_value > 5 else "B (false)"
        print(f"probe={probe_value}: branch {taken}, OUT_A.sum()={OUT_A.sum()}, OUT_B.sum()={OUT_B.sum()}")


if __name__ == "__main__":
    main()
