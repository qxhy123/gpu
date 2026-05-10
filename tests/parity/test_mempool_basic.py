import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_basic"


def test_mempool_basic_reuse_rate():
    """4 alloc/free cycles → first grows pool, next 3 reuse same block."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream
    from gpusim.trace.recorder import Recorder

    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()

    for _ in range(4):
        a = pool.malloc_async(s, 1024)
        a.buf[:] = 9
        pool.free_async(s, a)

    # 4 allocates: 1 fresh + 3 reused
    assert len(rec.pool_allocate_events) == 4
    reused = sum(1 for ev in rec.pool_allocate_events if ev.reused)
    assert reused == 3
    assert len(pool.slabs) == 1


def test_mempool_basic_runs():
    """run.py exits 0."""
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
