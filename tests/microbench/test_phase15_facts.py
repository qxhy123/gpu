"""Phase 15 microbench — stream capture + conditional/while node facts."""


def test_capture_appends_nodes_without_executing():
    """begin_capture → launch puts node in graph but does not execute kernel."""
    import numpy as np
    from gpusim.api import Stream
    from gpusim.config.loader import load_default
    cfg = load_default()
    ptx = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    OUT = np.zeros(32, dtype=np.uint32)
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": OUT}, kernel_name="t", config=cfg)
    s.end_capture()
    assert OUT.sum() == 0    # not executed during capture


def test_captured_graph_replay_equivalent_to_imperative():
    """Captured graph + replay produces same OUT as direct stream launch."""
    import numpy as np
    from gpusim.api import Stream, synchronize
    from gpusim.config.loader import load_default
    cfg = load_default()
    ptx = """
.visible .entry inc(.param .u64 OUT) {
    .reg .u64 %rd<5>; .reg .u32 %r<5>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    ld.global.u32 %r2, [%rd2]; add.u32 %r3, %r2, 1; st.global.u32 [%rd2], %r3;
    ret;
}
"""
    # Imperative
    OUT_imp = np.zeros(32, dtype=np.uint32)
    s_imp = Stream()
    s_imp.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT_imp}, kernel_name="t", config=cfg)
    synchronize(streams=[s_imp], config=cfg)
    # Captured + replay
    OUT_cap = np.zeros(32, dtype=np.uint32)
    s_cap = Stream()
    s_cap.begin_capture()
    s_cap.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT_cap}, kernel_name="t", config=cfg)
    g = s_cap.end_capture()
    g.instantiate(cfg).launch()
    assert OUT_imp.sum() == OUT_cap.sum() == 32


def test_conditional_branch_event_records_taken():
    """ConditionalBranch.taken matches cond_fn return value."""
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default
    cfg = load_default()
    rec = Recorder()
    g = Graph()
    g.add_conditional_node(cond_fn=lambda: False,
                              true_graph=Graph(), false_graph=Graph())
    exec = GraphExec.from_graph(g, cfg)
    exec._recorder = rec
    exec.launch()
    assert rec.conditional_branch_events[0].taken is False


def test_while_max_iterations_enforced():
    """Loop with always-True cond_fn raises after max_iterations."""
    import pytest
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_while_node(cond_fn=lambda: True, body_graph=Graph(), max_iterations=3)
    with pytest.raises(RuntimeError, match="exceeded max_iterations"):
        g.instantiate(cfg).launch()


def test_captured_graph_is_captured_flag_true():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    g = s.end_capture()
    assert g.is_captured is True


def test_handbuilt_graph_is_captured_flag_false():
    from gpusim.graph.graph import Graph
    g = Graph()
    assert g.is_captured is False
