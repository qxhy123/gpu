import numpy as np
import pathlib
import gpusim


def main():
    n_lo = 32 * 2     # 2 CTAs, 32 threads each
    n_hi = 32 * 64    # 64 CTAs, 32 threads each (heavy)
    a = np.arange(max(n_lo, n_hi), dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    print("# low concurrency (2 CTAs, 1 warp each):")
    out = np.zeros(n_lo, dtype=np.float32)
    res = gpusim.run(ptx_src=ptx, grid=(2, 1, 1), block=(32, 1, 1),
                     params={"A": a[:n_lo], "OUT": out}, mode="timing")
    print(f"  cycles={res.metrics['cycles']}")
    print("# high concurrency (64 CTAs, 1 warp each):")
    out = np.zeros(n_hi, dtype=np.float32)
    res = gpusim.run(ptx_src=ptx, grid=(64, 1, 1), block=(32, 1, 1),
                     params={"A": a[:n_hi], "OUT": out}, mode="timing")
    print(f"  cycles={res.metrics['cycles']}")


if __name__ == "__main__":
    main()
