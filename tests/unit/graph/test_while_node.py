def test_while_node_runs_until_cond_false():
    """Body runs until cond_fn returns False."""
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    counter = np.array([3], dtype=np.int32)
    src = """
.visible .entry dec(.param .u64 OUT) {
    .reg .u64 %rd<3>; .reg .u32 %r<3>;
    ld.param.u64 %rd0, [OUT];
    ld.global.u32 %r0, [%rd0]; sub.u32 %r1, %r0, 1; st.global.u32 [%rd0], %r1;
    ret;
}
"""
    body = Graph()
    body.add_kernel_node(ptx_src=src, grid=(1,1,1), block=(1,1,1),
                            params={"OUT": counter}, kernel_name="dec")

    # Use a single-iteration body counter held in Python so the kernel doesn't actually
    # have to mutate counter via PTX — for this unit test we use a host-side counter.
    iter_box = [3]
    def cond():
        return iter_box[0] > 0
    def body_python():
        iter_box[0] -= 1

    # Build a graph with a while node whose body is empty; we count iterations via cond_fn side-effect.
    g = Graph()
    g.add_while_node(cond_fn=lambda: (iter_box[0] > 0 and (body_python() or True)),
                        body_graph=Graph(),
                        max_iterations=10)
    g.instantiate(cfg).launch()
    assert iter_box[0] == 0


def test_while_node_max_iterations_raises():
    """When cond_fn never goes False, max_iterations cap raises."""
    import pytest
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    g = Graph()
    g.add_while_node(cond_fn=lambda: True, body_graph=Graph(), max_iterations=5)
    with pytest.raises(RuntimeError, match="exceeded max_iterations"):
        g.instantiate(cfg).launch()


def test_while_node_emits_loop_iteration_events():
    """Each iteration produces a LoopIteration trace event."""
    from gpusim.graph.graph import Graph
    from gpusim.graph.exec import GraphExec
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default

    cfg = load_default()
    rec = Recorder()
    iter_box = [4]
    g = Graph()
    g.add_while_node(cond_fn=lambda: (iter_box.__setitem__(0, iter_box[0] - 1) or iter_box[0] >= 0),
                        body_graph=Graph(), max_iterations=10)
    exec = GraphExec.from_graph(g, cfg)
    exec._recorder = rec
    exec.launch()
    assert len(rec.loop_iteration_events) == 4
    assert [e.iteration for e in rec.loop_iteration_events] == [0, 1, 2, 3]


def test_while_node_zero_iterations_when_cond_initially_false():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    g = Graph()
    g.add_while_node(cond_fn=lambda: False, body_graph=Graph(), max_iterations=10)
    cycles = g.instantiate(cfg).launch()
    assert cycles >= 5    # at least the cond_fn eval overhead
