"""Phase 13 metrics tests: graph_child_depth + graph_update_count."""


def test_graph_child_depth_no_children():
    from gpusim.analysis.metrics import graph_child_depth
    from gpusim.graph.graph import Graph
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1, 1, 1), block=(32, 1, 1),
                      params={}, kernel_name="k")
    assert graph_child_depth(g) == 0


def test_graph_child_depth_one_level():
    from gpusim.analysis.metrics import graph_child_depth
    from gpusim.graph.graph import Graph
    inner = Graph()
    inner.add_kernel_node(ptx_src="x", grid=(1, 1, 1), block=(32, 1, 1),
                          params={}, kernel_name="k_inner")
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    assert graph_child_depth(outer) == 1


def test_graph_child_depth_two_levels():
    from gpusim.analysis.metrics import graph_child_depth
    from gpusim.graph.graph import Graph
    inner_inner = Graph()
    inner_inner.add_kernel_node(ptx_src="x", grid=(1, 1, 1), block=(32, 1, 1),
                                params={}, kernel_name="k")
    inner = Graph()
    inner.add_child_graph_node(graph=inner_inner)
    outer = Graph()
    outer.add_child_graph_node(graph=inner)
    assert graph_child_depth(outer) == 2


def test_graph_update_count():
    from gpusim.analysis.metrics import graph_update_count
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1, 1, 1), block=(32, 1, 1),
                      params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    assert graph_update_count(exec) == 0
    exec.update_kernel_node_params(0, kernel_name="k2")
    assert graph_update_count(exec) == 1
