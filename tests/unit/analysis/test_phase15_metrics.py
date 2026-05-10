def test_stream_capture_count_zero_for_empty_trace():
    from gpusim.analysis.metrics import stream_capture_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    assert stream_capture_count(rec) == 0


def test_stream_capture_count_counts_end_events():
    from gpusim.analysis.metrics import stream_capture_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_end(stream_id=0, cycle=10, captured_node_count=3)
    rec.stream_capture_end(stream_id=1, cycle=20, captured_node_count=5)
    assert stream_capture_count(rec) == 2


def test_captured_node_count_sums_across_captures():
    from gpusim.analysis.metrics import captured_node_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.stream_capture_end(stream_id=0, cycle=10, captured_node_count=3)
    rec.stream_capture_end(stream_id=1, cycle=20, captured_node_count=5)
    assert captured_node_count(rec) == 8


def test_captured_node_count_zero_when_no_captures():
    from gpusim.analysis.metrics import captured_node_count
    from gpusim.trace.recorder import Recorder
    assert captured_node_count(Recorder()) == 0


def test_conditional_branch_taken_count_only_counts_true():
    from gpusim.analysis.metrics import conditional_branch_taken_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.conditional_branch(node_id=0, taken=True, cycle=0)
    rec.conditional_branch(node_id=1, taken=False, cycle=10)
    rec.conditional_branch(node_id=2, taken=True, cycle=20)
    assert conditional_branch_taken_count(rec) == 2


def test_conditional_branch_taken_count_zero_when_none_taken():
    from gpusim.analysis.metrics import conditional_branch_taken_count
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    rec.conditional_branch(node_id=0, taken=False, cycle=0)
    assert conditional_branch_taken_count(rec) == 0


def test_avg_loop_iterations_returns_zero_for_no_while_nodes():
    from gpusim.analysis.metrics import avg_loop_iterations
    from gpusim.trace.recorder import Recorder
    assert avg_loop_iterations(Recorder()) == 0.0


def test_avg_loop_iterations_per_node():
    """node_id 0 ran 3 iterations, node_id 1 ran 5 — average is 4.0."""
    from gpusim.analysis.metrics import avg_loop_iterations
    from gpusim.trace.recorder import Recorder
    rec = Recorder()
    for i in range(3):
        rec.loop_iteration(node_id=0, iteration=i, cycle=i*10)
    for i in range(5):
        rec.loop_iteration(node_id=1, iteration=i, cycle=100+i*10)
    assert avg_loop_iterations(rec) == 4.0
