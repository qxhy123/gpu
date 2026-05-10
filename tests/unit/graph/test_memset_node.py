def test_memset_node_args():
    import numpy as np
    from gpusim.graph.node import MemsetNodeArgs
    buf = np.zeros(8, dtype=np.uint8)
    a = MemsetNodeArgs(buf=buf, value=0xff, n_bytes=8)
    assert a.value == 0xff


def test_graph_add_memset_node():
    import numpy as np
    from gpusim.graph.graph import Graph
    g = Graph()
    buf = np.zeros(8, dtype=np.uint8)
    nid = g.add_memset_node(buf=buf, value=42, n_bytes=8)
    assert nid == 0
    assert g.nodes[0].type == "memset"
    assert g.nodes[0].memset_args.value == 42


def test_graph_exec_memset_fills_buffer():
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    buf = np.full(8, 1, dtype=np.uint8)
    g = Graph()
    g.add_memset_node(buf=buf, value=0, n_bytes=8)
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    np.testing.assert_array_equal(buf, np.zeros(8, dtype=np.uint8))
    assert cycles == 50    # fixed memset overhead
