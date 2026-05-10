def test_graph_is_captured_default_false():
    from gpusim.graph.graph import Graph
    g = Graph()
    assert g.is_captured is False


def test_graph_is_captured_can_be_set():
    from gpusim.graph.graph import Graph
    g = Graph()
    g.is_captured = True
    assert g.is_captured is True


def test_capture_session_creates_shared_graph():
    from gpusim.graph.capture_session import CaptureSession
    sess = CaptureSession()
    assert sess.graph is not None
    assert sess.graph.is_captured is True
    assert len(sess.streams) == 0


def test_capture_session_attach_stream_records_membership():
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream
    sess = CaptureSession()
    s = Stream()
    sess.attach(s)
    assert s in sess.streams
    assert s._captured_graph is sess.graph
    assert s._capture_session is sess


def test_capture_session_attach_double_raises():
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream
    sess = CaptureSession()
    s = Stream()
    sess.attach(s)
    import pytest
    with pytest.raises(RuntimeError, match="already attached"):
        sess.attach(s)


def test_capture_session_end_returns_graph_and_detaches_streams():
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream
    sess = CaptureSession()
    s1 = Stream()
    s2 = Stream()
    sess.attach(s1)
    sess.attach(s2)
    g = sess.end()
    assert g is sess.graph
    assert s1._captured_graph is None
    assert s2._captured_graph is None
    assert s1._capture_session is None
    assert s2._capture_session is None


def test_session_cross_stream_record_then_wait_creates_edge():
    """sA.record(ev) → sB.wait(ev) should create a graph edge from sA's record node to sB's wait node."""
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream, Event
    sess = CaptureSession()
    sA = Stream()
    sB = Stream()
    sess.attach(sA)
    sess.attach(sB)
    ev = Event()
    sA.record(ev)        # adds event node, registers event_id -> node_id in session
    sB.wait(ev)          # adds event node + creates cross-stream edge
    g = sess.end()
    assert len(g.nodes) == 2
    record_node = next(n for n in g.nodes if n.event_args.op == "record")
    wait_node = next(n for n in g.nodes if n.event_args.op == "wait")
    assert (record_node.node_id, wait_node.node_id) in g.edges


def test_session_wait_for_unrecorded_event_no_edge():
    """If the event was never recorded inside the session, wait creates a node but no cross-edge."""
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream, Event
    sess = CaptureSession()
    sA = Stream()
    sess.attach(sA)
    ev = Event()
    sA.wait(ev)          # never recorded
    g = sess.end()
    assert len(g.nodes) == 1
    # no cross-edge (only intra-stream chaining edges, of which there are none here)
    assert g.edges == []
