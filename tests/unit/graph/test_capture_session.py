def test_graph_is_captured_default_false():
    from gpusim.graph.graph import Graph
    g = Graph()
    assert g.is_captured is False


def test_graph_is_captured_can_be_set():
    from gpusim.graph.graph import Graph
    g = Graph()
    g.is_captured = True
    assert g.is_captured is True
