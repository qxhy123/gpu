import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_graph_with_child_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_with_child"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_update_replay_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_update_replay"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
