import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_graph_explicit_build_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_explicit_build"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_graph_replay_perf_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_replay_perf"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
