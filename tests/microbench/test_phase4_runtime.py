import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_multi_sm_scheduler_runtime_under_60s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_sm_scheduler"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=120)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 60, f"multi_sm_scheduler took {elapsed:.1f}s (limit 60s)"
