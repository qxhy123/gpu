import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_conditional_branch"


def test_graph_conditional_branch_correctness():
    """Probe buffer determines which branch runs."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    OUT_A = np.zeros(32, dtype=np.uint32)
    OUT_B = np.zeros(32, dtype=np.uint32)
    probe = np.array([10], dtype=np.int32)    # > 5 → take true branch (A)

    g_A = Graph()
    g_A.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": OUT_A}, kernel_name="A")
    g_B = Graph()
    g_B.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": OUT_B}, kernel_name="B")

    g = Graph()
    g.add_conditional_node(cond_fn=lambda: probe[0] > 5,
                              true_graph=g_A, false_graph=g_B)
    g.instantiate(cfg).launch()
    assert OUT_A.sum() == 32
    assert OUT_B.sum() == 0

    # flip probe, run again on fresh buffers
    OUT_A2 = np.zeros(32, dtype=np.uint32)
    OUT_B2 = np.zeros(32, dtype=np.uint32)
    probe2 = np.array([2], dtype=np.int32)    # <= 5 -> take false branch (B)

    g_A2 = Graph()
    g_A2.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_A2}, kernel_name="A")
    g_B2 = Graph()
    g_B2.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_B2}, kernel_name="B")

    g2 = Graph()
    g2.add_conditional_node(cond_fn=lambda: probe2[0] > 5,
                               true_graph=g_A2, false_graph=g_B2)
    g2.instantiate(cfg).launch()
    assert OUT_A2.sum() == 0
    assert OUT_B2.sum() == 32
