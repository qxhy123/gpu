import pathlib


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_fragmentation"


def test_mempool_fragmentation_best_fit_reuses_correct_blocks():
    """After freeing all, re-alloc 1024 + 2048 reuses exact size matches."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream

    pool = MemoryPool()
    s = Stream()
    a1024 = pool.malloc_async(s, 1024)
    a2048 = pool.malloc_async(s, 2048)
    a4096 = pool.malloc_async(s, 4096)
    a1024b = pool.malloc_async(s, 1024)
    a2048b = pool.malloc_async(s, 2048)
    assert len(pool.slabs) == 5

    for a in (a1024, a2048, a4096, a1024b, a2048b):
        pool.free_async(s, a)

    # Re-alloc 1024 → reuses one of the 1024 blocks (oldest freed)
    new1024 = pool.malloc_async(s, 1024)
    assert new1024._slab_index == a1024._slab_index    # oldest 1024 wins

    # Re-alloc 2048 → reuses oldest 2048
    new2048 = pool.malloc_async(s, 2048)
    assert new2048._slab_index == a2048._slab_index

    assert len(pool.slabs) == 5    # no new growth


def test_mempool_fragmentation_runs():
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
