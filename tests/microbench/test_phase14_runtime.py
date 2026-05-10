import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_persistent_kernel_server_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "persistent_kernel_server"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_dynamic_parallelism_recursive_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "dynamic_parallelism_recursive"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
