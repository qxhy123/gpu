"""Tests for topological sort + cycle detection (T3)."""
import pytest


def _make_node(node_id):
    from gpusim.graph.node import GraphNode, KernelNodeArgs
    args = KernelNodeArgs(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                          params={}, kernel_name=f"k{node_id}")
    return GraphNode(node_id=node_id, type="kernel", kernel_args=args)


def test_topological_sort_linear_chain():
    """Linear chain 0->1->2 should produce [0, 1, 2]."""
    from gpusim.graph.exec import _topological_sort
    nodes = [_make_node(i) for i in range(3)]
    edges = [(0, 1), (1, 2)]
    order = _topological_sort(nodes, edges)
    # 0 must come before 1, 1 must come before 2
    assert order.index(0) < order.index(1)
    assert order.index(1) < order.index(2)


def test_topological_sort_diamond():
    """Diamond: 0->{1,2}->3. 0 first, 3 last; 1 and 2 in middle."""
    from gpusim.graph.exec import _topological_sort
    nodes = [_make_node(i) for i in range(4)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    order = _topological_sort(nodes, edges)
    assert order[0] == 0
    assert order[-1] == 3
    assert set(order[1:3]) == {1, 2}


def test_topological_sort_cycle_raises():
    """Cycle 0->1->2->0 must raise ValueError."""
    from gpusim.graph.exec import _topological_sort
    nodes = [_make_node(i) for i in range(3)]
    edges = [(0, 1), (1, 2), (2, 0)]
    with pytest.raises(ValueError, match="cycle"):
        _topological_sort(nodes, edges)
