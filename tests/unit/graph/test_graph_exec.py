"""T6 + T7: GraphExec.launch tests (single kernel + 3-kernel chain)."""
from __future__ import annotations

import numpy as np


def test_graph_exec_from_graph():
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src=".entry t() { .reg .u32 %r0; mov.u32 %r0, %tid.x; ret; }",
                      grid=(1, 1, 1), block=(32, 1, 1), params={}, kernel_name="k")
    exec = GraphExec.from_graph(g, cfg)
    assert exec.topo_order == [0]


def test_graph_exec_launch_single_kernel():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    g = Graph()
    g.add_kernel_node(ptx_src=src, grid=(1, 1, 1), block=(32, 1, 1),
                      params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    cycles = exec.launch()
    assert cycles > 0


def test_graph_exec_chain_3_kernels():
    """Build A -> B -> C dependency chain; verify launch executes all 3."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)

    src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    ret;
}
"""
    g = Graph()
    n0 = g.add_kernel_node(ptx_src=src, grid=(1, 1, 1), block=(32, 1, 1),
                           params={"A": A, "B": B, "OUT": OUT},
                           kernel_name="vec_add_0")
    n1 = g.add_kernel_node(ptx_src=src, grid=(1, 1, 1), block=(32, 1, 1),
                           params={"A": A, "B": B, "OUT": OUT},
                           kernel_name="vec_add_1")
    n2 = g.add_kernel_node(ptx_src=src, grid=(1, 1, 1), block=(32, 1, 1),
                           params={"A": A, "B": B, "OUT": OUT},
                           kernel_name="vec_add_2")
    g.add_dependency(n0, n1)
    g.add_dependency(n1, n2)

    exec = g.instantiate(cfg)
    cycles = exec.launch()

    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0


def test_graph_exec_records_launch_event():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    g = Graph()
    g.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    rec = Recorder()
    exec = g.instantiate(cfg)
    exec._recorder = rec
    exec._graph_id = 0
    exec.launch()
    assert len(rec.graph_launch_events) == 1
