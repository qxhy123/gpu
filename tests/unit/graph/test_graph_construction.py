"""Tests for Graph class: construction, add_kernel, add_dependency (T2)."""
import pytest


def test_graph_empty_construction():
    from gpusim.graph.graph import Graph
    g = Graph()
    assert g.nodes == []
    assert g.edges == []


def test_graph_add_kernel_sequential_ids():
    from gpusim.graph.graph import Graph
    g = Graph()
    id0 = g.add_kernel_node(ptx_src="a", grid=(1,1,1), block=(32,1,1),
                             params={}, kernel_name="k0")
    id1 = g.add_kernel_node(ptx_src="b", grid=(1,1,1), block=(32,1,1),
                             params={}, kernel_name="k1")
    id2 = g.add_kernel_node(ptx_src="c", grid=(1,1,1), block=(32,1,1),
                             params={}, kernel_name="k2")
    assert id0 == 0
    assert id1 == 1
    assert id2 == 2
    assert len(g.nodes) == 3


def test_graph_add_dependency():
    from gpusim.graph.graph import Graph
    g = Graph()
    a = g.add_kernel_node(ptx_src="a", grid=(1,1,1), block=(32,1,1),
                           params={}, kernel_name="a")
    b = g.add_kernel_node(ptx_src="b", grid=(1,1,1), block=(32,1,1),
                           params={}, kernel_name="b")
    g.add_dependency(a, b)
    assert (0, 1) in g.edges


def test_graph_self_dependency_raises():
    from gpusim.graph.graph import Graph
    g = Graph()
    nid = g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                             params={}, kernel_name="x")
    with pytest.raises(ValueError, match="self-dependency"):
        g.add_dependency(nid, nid)
