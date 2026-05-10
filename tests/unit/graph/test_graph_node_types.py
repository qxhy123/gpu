"""Tests for GraphNode + 3 args dataclasses (T1) and memcpy/event node coverage (T4)."""


def test_kernel_node_args():
    from gpusim.graph.node import KernelNodeArgs
    a = KernelNodeArgs(ptx_src=".entry t() { ret; }",
                          grid=(1,1,1), block=(32,1,1),
                          params={}, kernel_name="k")
    assert a.kernel_name == "k"
    assert a.grid == (1,1,1)


def test_memcpy_node_args():
    import numpy as np
    from gpusim.graph.node import MemcpyNodeArgs
    src = np.zeros(8); dst = np.zeros(8)
    a = MemcpyNodeArgs(src=src, dst=dst, n_bytes=64)
    assert a.n_bytes == 64


def test_graph_node_kernel_type():
    from gpusim.graph.node import GraphNode, KernelNodeArgs
    args = KernelNodeArgs(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="k")
    n = GraphNode(node_id=0, type="kernel", kernel_args=args)
    assert n.type == "kernel"
    assert n.kernel_args is args


def test_graph_add_memcpy_node():
    import numpy as np
    from gpusim.graph.graph import Graph
    src = np.zeros(8, dtype=np.float32)
    dst = np.zeros(8, dtype=np.float32)
    g = Graph()
    nid = g.add_memcpy_node(src=src, dst=dst, n_bytes=32)
    assert nid == 0
    assert g.nodes[0].type == "memcpy"
    assert g.nodes[0].memcpy_args.n_bytes == 32


def test_graph_add_event_node():
    from gpusim.graph.graph import Graph
    from gpusim.api import Event
    g = Graph()
    ev = Event()
    nid = g.add_event_node(event=ev, op="record")
    assert g.nodes[0].type == "event"
    assert g.nodes[0].event_args.op == "record"
    assert g.nodes[0].event_args.event is ev
