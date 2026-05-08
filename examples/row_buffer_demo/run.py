import numpy as np
import pathlib
import gpusim


def main():
    n = 16 << 20  # 16 MB float array — large enough for all stride values
    a = np.arange(n, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    # stride=32: 32 threads × 32 elements apart = 32 distinct 128 B cache lines,
    # all in row 0 of their respective channels. Each channel serves row 0 first
    # (ROW_MISS), then subsequent accesses to the same open row are ROW_HIT.
    print("# stride=32 (within-row, row hits dominate):")
    out = np.zeros(32, dtype=np.float32)
    res = gpusim.run(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                     params={"A": a, "OUT": out, "STRIDE": 32}, mode="timing")
    print(f"  cycles={res.metrics['cycles']}, "
          f"row_buffer_hit_rate={res.cache_metrics['row_buffer_hit_rate']:.2%}")

    # stride=65568: each thread jumps to a completely different HBM row
    # (= 65568 floats = 262 272 bytes = line_addr increment of 2049, which changes
    # the HBM row field AND maps to different L1 sets, avoiding cache-set thrashing).
    # Every access opens a new row → all ROW_MISS.
    print("# stride=65568 (cross-row stride, row misses dominate):")
    out = np.zeros(32, dtype=np.float32)
    res = gpusim.run(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                     params={"A": a, "OUT": out, "STRIDE": 65568}, mode="timing")
    print(f"  cycles={res.metrics['cycles']}, "
          f"row_buffer_hit_rate={res.cache_metrics['row_buffer_hit_rate']:.2%}")


if __name__ == "__main__":
    main()
