import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_stream_capture_basic_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_basic"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_stream_capture_multi_stream_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_multi_stream"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_conditional_branch_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_conditional_branch"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_while_loop_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_while_loop"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
