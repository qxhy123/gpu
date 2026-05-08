"""Phase 2 microbench assertions: cache hit rates, bandwidth, row-buffer locality."""
import numpy as np
import pathlib
import gpusim


_L1_THRASH_PTX = (
    pathlib.Path(__file__).parents[2] / "examples/l1_thrash_demo/kernel.ptx"
).read_text()

_ROW_BUFFER_PTX = (
    pathlib.Path(__file__).parents[2] / "examples/row_buffer_demo/kernel.ptx"
).read_text()


def test_data_fits_l1_high_l1_hit_rate():
    """Working set fitting in L1 with reuse → L1 hit rate >= 0.5 after warmup.

    Use STRIDE=1 so all 32 threads access a 1-element stride window; the same
    cache lines are accessed on every iteration, giving high L1 hit rate.
    """
    n = 16 << 20
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(32, dtype=np.float32)
    res = gpusim.run(
        ptx_src=_L1_THRASH_PTX,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        params={"A": a, "OUT": out, "K": 32, "STRIDE": 1},
        mode="timing",
    )
    assert res.cache_metrics["l1_hit_rate"] >= 0.5, (
        f"Expected l1_hit_rate >= 0.5 for STRIDE=1 (reusing same lines), "
        f"got {res.cache_metrics['l1_hit_rate']:.4f}"
    )


def test_strided_access_low_row_buffer_hit_rate():
    """Large cross-row stride → nearly all HBM accesses are row misses.

    stride=65568 floats maps consecutive threads to different HBM rows and
    different L1 sets (avoids cache-set thrashing while still causing row misses).
    """
    n = 16 << 20
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(32, dtype=np.float32)
    res = gpusim.run(
        ptx_src=_ROW_BUFFER_PTX,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        params={"A": a, "OUT": out, "STRIDE": 65568},
        mode="timing",
    )
    assert res.cache_metrics["row_buffer_hit_rate"] <= 0.2, (
        f"Expected row_buffer_hit_rate <= 0.2 for cross-row stride, "
        f"got {res.cache_metrics['row_buffer_hit_rate']:.4f}"
    )


def test_sequential_access_high_row_buffer_hit_rate():
    """Within-row stride → majority of HBM accesses are row hits.

    stride=32 (32 floats = 1 cache line apart) puts each thread in a different
    128 B line, all within row 0 of their respective channels. After the first
    per-channel access opens the row, subsequent accesses to the same open row
    are ROW_HIT.
    """
    n = 16 << 20
    a = np.arange(n, dtype=np.float32)
    out = np.zeros(32, dtype=np.float32)
    res = gpusim.run(
        ptx_src=_ROW_BUFFER_PTX,
        grid=(1, 1, 1),
        block=(32, 1, 1),
        params={"A": a, "OUT": out, "STRIDE": 32},
        mode="timing",
    )
    assert res.cache_metrics["row_buffer_hit_rate"] >= 0.5, (
        f"Expected row_buffer_hit_rate >= 0.5 for within-row stride, "
        f"got {res.cache_metrics['row_buffer_hit_rate']:.4f}"
    )
