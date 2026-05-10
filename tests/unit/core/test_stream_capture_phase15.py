import pytest


def test_begin_capture_default_mode_global_works():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()                    # default mode="global"
    g = s.end_capture()
    assert g is not None


def test_begin_capture_explicit_global_works():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture(mode="global")
    g = s.end_capture()
    assert g is not None


def test_begin_capture_unknown_mode_raises():
    from gpusim.api import Stream
    s = Stream()
    with pytest.raises(ValueError, match="only 'global' capture mode supported"):
        s.begin_capture(mode="thread")


def test_begin_capture_double_begin_raises():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    with pytest.raises(RuntimeError, match="already capturing"):
        s.begin_capture()


def test_end_capture_without_begin_raises():
    from gpusim.api import Stream
    s = Stream()
    with pytest.raises(RuntimeError, match="not capturing"):
        s.end_capture()


def test_end_capture_marks_graph_is_captured():
    from gpusim.api import Stream
    s = Stream()
    s.begin_capture()
    g = s.end_capture()
    assert g.is_captured is True


def test_recorder_records_stream_capture_begin():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_begin(stream_id=3, cycle=0)
    assert len(rec.stream_capture_begin_events) == 1
    ev = rec.stream_capture_begin_events[0]
    assert ev.stream_id == 3
    assert ev.cycle == 0


def test_recorder_records_stream_capture_end():
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_end(stream_id=3, cycle=10, captured_node_count=5)
    assert len(rec.stream_capture_end_events) == 1
    ev = rec.stream_capture_end_events[0]
    assert ev.stream_id == 3
    assert ev.cycle == 10
    assert ev.captured_node_count == 5


def test_stream_begin_capture_with_recorder_emits_begin_event():
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    s = Stream()
    s._recorder = rec
    s.begin_capture()
    assert len(rec.stream_capture_begin_events) == 1
    assert rec.stream_capture_begin_events[0].stream_id == s.stream_id


def test_stream_end_capture_with_recorder_emits_end_event_with_node_count():
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    s = Stream()
    s._recorder = rec
    s.begin_capture()
    s.end_capture()
    assert len(rec.stream_capture_end_events) == 1
    assert rec.stream_capture_end_events[0].captured_node_count == 0


def test_capture_records_event_node_for_record():
    from gpusim.api import Stream, Event
    s = Stream()
    ev = Event()
    s.begin_capture()
    s.record(ev)
    g = s.end_capture()
    assert len(g.nodes) == 1
    assert g.nodes[0].type == "event"
    assert g.nodes[0].event_args.op == "record"
    assert g.nodes[0].event_args.event is ev


def test_capture_records_event_node_for_wait():
    from gpusim.api import Stream, Event
    s = Stream()
    ev = Event()
    s.begin_capture()
    s.wait(ev)
    g = s.end_capture()
    assert len(g.nodes) == 1
    assert g.nodes[0].type == "event"
    assert g.nodes[0].event_args.op == "wait"


def test_capture_chains_kernel_to_record_to_kernel_with_edges():
    """Within a single stream, ordering is captured as edges."""
    from gpusim.api import Stream, Event
    from gpusim.config.loader import load_default
    cfg = load_default()
    ptx = """
.visible .entry k(.param .u64 OUT) {
    .reg .u64 %rd<3>; .reg .u32 %r<3>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    import numpy as np
    OUT = np.zeros(32, dtype=np.uint32)
    s = Stream()
    ev = Event()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": OUT}, kernel_name="k1", config=cfg)
    s.record(ev)
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": OUT}, kernel_name="k2", config=cfg)
    g = s.end_capture()
    assert len(g.nodes) == 3                    # k1, record, k2
    assert len(g.edges) == 2                    # k1→record, record→k2
    nids = [n.node_id for n in g.nodes]
    assert (nids[0], nids[1]) in g.edges
    assert (nids[1], nids[2]) in g.edges
