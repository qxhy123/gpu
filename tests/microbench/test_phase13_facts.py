"""Phase 13 microbench — graphs completion facts."""


def test_memset_node_cycles_50():
    """Memset node = 50 cycles."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    buf = np.full(8, 1, dtype=np.uint8)
    g = Graph()
    g.add_memset_node(buf=buf, value=0, n_bytes=8)
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    assert cycles == 50


def test_child_depth_3_levels():
    """3-level nested graph depth = 3."""
    from gpusim.graph.graph import Graph
    from gpusim.analysis.metrics import graph_child_depth
    g3 = Graph()
    g3.add_kernel_node(ptx_src="x", grid=(1, 1, 1), block=(32, 1, 1),
                       params={}, kernel_name="leaf")
    g2 = Graph()
    g2.add_child_graph_node(graph=g3)
    g1 = Graph()
    g1.add_child_graph_node(graph=g2)
    g0 = Graph()
    g0.add_child_graph_node(graph=g1)
    assert graph_child_depth(g0) == 3


def test_update_count_tracks_calls():
    """Update count increments per call."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1, 1, 1), block=(32, 1, 1),
                      params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    for _ in range(5):
        exec.update_kernel_node_params(0, kernel_name="k_new")
    assert exec._update_count == 5
