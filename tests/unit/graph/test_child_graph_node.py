def test_child_graph_node_args():
    from gpusim.graph.node import ChildGraphNodeArgs
    from gpusim.graph.graph import Graph
    inner = Graph()
    a = ChildGraphNodeArgs(graph=inner)
    assert a.graph is inner


def test_graph_add_child_graph_node():
    from gpusim.graph.graph import Graph
    inner = Graph()
    inner.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                            params={}, kernel_name="inner_k")
    outer = Graph()
    nid = outer.add_child_graph_node(graph=inner)
    assert outer.nodes[0].type == "child_graph"
    assert outer.nodes[0].child_graph_args.graph is inner


def test_graph_exec_child_graph_executes():
    """Child graph nested execution produces correct outputs."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)

    src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>; .reg .u32 %r<4>; .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A]; ld.param.u64 %rd1, [B]; ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3; ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3; ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3; st.global.f32 [%rd4], %f2;
    ret;
}
"""
    inner = Graph()
    inner.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                            params={"A": A, "B": B, "OUT": OUT},
                            kernel_name="vec_add")

    outer = Graph()
    outer.add_child_graph_node(graph=inner)

    exec = outer.instantiate(cfg)
    cycles = exec.launch()

    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
