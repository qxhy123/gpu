import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_ring_allreduce_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "ring_allreduce"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_ddp_training_step_runtime_under_60s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "ddp_training_step"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=120)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 60
