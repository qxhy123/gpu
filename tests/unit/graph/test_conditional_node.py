def test_add_conditional_node_appends():
    from gpusim.graph.graph import Graph
    g_outer = Graph()
    g_true = Graph()
    g_false = Graph()
    nid = g_outer.add_conditional_node(
        cond_fn=lambda: True,
        true_graph=g_true,
        false_graph=g_false,
    )
    assert isinstance(nid, int)
    assert len(g_outer.nodes) == 1
    node = g_outer.nodes[0]
    assert node.type == "conditional"
    assert node.conditional_args is not None
    assert node.conditional_args.true_graph is g_true
    assert node.conditional_args.false_graph is g_false


def test_conditional_args_stores_callable():
    from gpusim.graph.graph import Graph
    from gpusim.graph.node import ConditionalNodeArgs
    g = Graph()
    f = lambda: True
    nid = g.add_conditional_node(cond_fn=f, true_graph=Graph(), false_graph=Graph())
    args = g.nodes[0].conditional_args
    assert isinstance(args, ConditionalNodeArgs)
    assert args.cond_fn is f
