import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_reduce_scatter_fsdp_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "reduce_scatter_fsdp"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_pytorch_dist_simple_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "pytorch_dist_simple"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                          capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
