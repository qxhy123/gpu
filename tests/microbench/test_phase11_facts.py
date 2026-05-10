"""Phase 11 microbench — CUDA Graphs facts."""


def test_topo_sort_known_graph():
    """Topological sort produces valid order for diamond graph."""
    from gpusim.graph.exec import _topological_sort
    from gpusim.graph.node import GraphNode
    nodes = [GraphNode(node_id=i, type="kernel") for i in range(4)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    order = _topological_sort(nodes, edges)
    assert order[0] == 0
    assert order[-1] == 3


def test_capture_chain_dependency_count():
    """3 capture launches → 2 implicit dependencies."""
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    for i in range(3):
        s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                  params={}, kernel_name=f"k{i}")
    g = s.end_capture()
    assert len(g.nodes) == 3
    assert len(g.edges) == 2


def test_graph_replay_deterministic():
    """Replay 3x produces same cycles."""
    import numpy as np
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    cfg = load_default()
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
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
    OUT = np.zeros(n, dtype=np.float32)
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT}, kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    cycles = [exec.launch() for _ in range(3)]
    assert cycles[0] == cycles[1] == cycles[2]
