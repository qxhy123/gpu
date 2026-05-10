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


def test_conditional_takes_true_branch():
    """When cond_fn returns True, only true_graph executes."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    src = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2]; add.u32 %r3, %r2, 1; st.global.u32 [%rd2], %r3;
    ret;
}
"""
    OUT_T = np.zeros(32, dtype=np.uint32)
    OUT_F = np.zeros(32, dtype=np.uint32)
    g_true = Graph()
    g_true.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_T}, kernel_name="t")
    g_false = Graph()
    g_false.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                               params={"OUT": OUT_F}, kernel_name="f")

    g_outer = Graph()
    g_outer.add_conditional_node(cond_fn=lambda: True,
                                    true_graph=g_true, false_graph=g_false)
    g_outer.instantiate(cfg).launch()
    assert OUT_T.sum() == 32
    assert OUT_F.sum() == 0


def test_conditional_takes_false_branch():
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    src = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2]; add.u32 %r3, %r2, 1; st.global.u32 [%rd2], %r3;
    ret;
}
"""
    OUT_T = np.zeros(32, dtype=np.uint32)
    OUT_F = np.zeros(32, dtype=np.uint32)
    g_true = Graph()
    g_true.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                              params={"OUT": OUT_T}, kernel_name="t")
    g_false = Graph()
    g_false.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                               params={"OUT": OUT_F}, kernel_name="f")

    g_outer = Graph()
    g_outer.add_conditional_node(cond_fn=lambda: False,
                                    true_graph=g_true, false_graph=g_false)
    g_outer.instantiate(cfg).launch()
    assert OUT_T.sum() == 0
    assert OUT_F.sum() == 32


def test_conditional_with_empty_false_branch():
    """If false_graph has no nodes, false branch executes 0 nodes (no error)."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    g_outer = Graph()
    g_outer.add_conditional_node(cond_fn=lambda: False,
                                    true_graph=Graph(), false_graph=Graph())
    cycles = g_outer.instantiate(cfg).launch()
    assert cycles >= 5    # at least the 5-cycle conditional eval overhead


def test_conditional_emits_trace_event():
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default

    cfg = load_default()
    rec = Recorder()
    g = Graph()
    g.add_conditional_node(cond_fn=lambda: True,
                              true_graph=Graph(), false_graph=Graph())
    exec = GraphExec.from_graph(g, cfg)
    exec._recorder = rec
    exec.launch()
    assert len(rec.conditional_branch_events) == 1
    ev = rec.conditional_branch_events[0]
    assert ev.taken is True
