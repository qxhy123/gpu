import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_memset_zero"


def test_graph_memset_zero_correctness():
    """Memset-zero -> kernel write -> memset-zero. Final buffer = zeros."""
    import gpusim
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    n = 32
    buf = np.full(n * 4, 99, dtype=np.uint8)   # n*4 bytes for n float32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)   # we'll repurpose buf for OUT

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    g = Graph()
    # Pre-zero buffer
    n0 = g.add_memset_node(buf=buf, value=0, n_bytes=n * 4)
    # Compute kernel writes to OUT (separate buffer)
    n1 = g.add_kernel_node(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                           params={"A": A, "B": B, "OUT": OUT},
                           kernel_name="vec_add")
    g.add_dependency(n0, n1)
    # Post-zero buf (different from OUT)
    n2 = g.add_memset_node(buf=buf, value=0, n_bytes=n * 4)
    g.add_dependency(n1, n2)

    exec = g.instantiate(cfg)
    cycles = exec.launch()

    np.testing.assert_array_equal(buf, np.zeros(n * 4, dtype=np.uint8))
    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 100   # 2 memsets (50 each) + kernel
