import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_phase8_overlap_real_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "phase8_overlap_real"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_event_timing_benchmark_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "event_timing_benchmark"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
