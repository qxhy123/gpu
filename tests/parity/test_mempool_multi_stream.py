import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "mempool_multi_stream"


def test_mempool_multi_stream_sync_promotes_block():
    """sA alloc/free + synchronize_stream → sB malloc reuses."""
    from gpusim.mempool.pool import MemoryPool
    from gpusim.api import Stream

    pool = MemoryPool()
    sA = Stream()
    sB = Stream()

    # Without synchronize_stream, sB grows the pool
    a = sA.malloc_async(pool, 256)
    sA.free_async(pool, a)
    sB.malloc_async(pool, 256)
    assert len(pool.slabs) == 2

    # Reset for the synchronized half
    pool2 = MemoryPool()
    sA2 = Stream()
    sB2 = Stream()
    a2 = sA2.malloc_async(pool2, 256)
    sA2.free_async(pool2, a2)
    pool2.synchronize_stream(sA2)
    sB2.malloc_async(pool2, 256)
    assert len(pool2.slabs) == 1


def test_mempool_multi_stream_runs():
    import subprocess, sys
    res = subprocess.run([sys.executable, str(_DIR / "run.py")],
                         capture_output=True, timeout=30)
    assert res.returncode == 0, res.stderr.decode()[-500:]
