import pytest, time, pathlib, subprocess, sys


@pytest.mark.slow
def test_mempool_basic_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_basic"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_mempool_multi_stream_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_multi_stream"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_mempool_fragmentation_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_fragmentation"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30


@pytest.mark.slow
def test_mempool_train_step_runtime_under_30s():
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_train_step"
    t0 = time.time()
    res = subprocess.run([sys.executable, str(base / "run.py")],
                         capture_output=True, timeout=60)
    elapsed = time.time() - t0
    assert res.returncode == 0
    assert elapsed < 30
