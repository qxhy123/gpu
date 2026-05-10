import pathlib


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_train_step"


def test_mempool_train_step_high_water_stabilizes():
    """Train-step pattern: each iter alloc + free; high_water == iter1 working set."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream

    pool = MemoryPool()
    s = Stream()

    for _ in range(10):
        act = s.malloc_async(pool, 8 * 1024)
        grad = s.malloc_async(pool, 8 * 1024)
        # ... in real training, kernel would touch these ...
        s.free_async(pool, act)
        s.free_async(pool, grad)
        pool.synchronize_stream(s)

    # high_water_mark should equal iter 1's peak (16 KB, two 8 KB blocks)
    assert pool.high_water_mark == 16 * 1024
    assert len(pool.slabs) == 2          # only 2 slabs, reused 9 more iters


def test_mempool_train_step_reuse_rate_high():
    """After 10 iters of alloc/free, reuse rate should be 18/20 = 0.9 (first 2 fresh, next 18 reused)."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.trace.recorder import Recorder
    from gpusim.api import Stream
    from gpusim.analysis.metrics import pool_reuse_rate, pool_alloc_count

    rec = Recorder()
    pool = MemoryPool()
    pool._recorder = rec
    s = Stream()

    for _ in range(10):
        act = s.malloc_async(pool, 8 * 1024)
        grad = s.malloc_async(pool, 8 * 1024)
        s.free_async(pool, act)
        s.free_async(pool, grad)

    assert pool_alloc_count(rec, pool._pool_id) == 20
    assert pool_reuse_rate(rec, pool._pool_id) == 0.9    # 18 reused / 20 total


def test_mempool_train_step_runs():
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
