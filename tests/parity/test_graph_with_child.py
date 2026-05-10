import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_with_child"


def test_graph_with_child_correctness():
    """Outer graph contains a child graph (with 2 kernel nodes)."""
    import gpusim
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    # Build child graph: 2 kernel nodes (chain)
    inner = Graph()
    n0 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_0")
    n1 = inner.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                                  params={"A": A, "B": B, "OUT": OUT},
                                  kernel_name="vec_add_inner_1")
    inner.add_dependency(n0, n1)

    # Build outer graph with one child node
    outer = Graph()
    outer.add_child_graph_node(graph=inner)

    exec = outer.instantiate(cfg)
    cycles = exec.launch()

    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
