import numpy as np, pathlib, gpusim


def main():
    n = 16 << 20
    a = np.arange(n, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    print("# Three working-set configurations (OUTER_LOOPS revisits the K-line set):")
    for label, K, STRIDE, OUTER in [
        ("A: fits L1 (~32 KB)",      256,   32, 8),
        ("B: > L1, fits L2 (~1 MB)", 8192,  32, 4),
        ("C: > L2 (~5 MB)",          40000, 32, 2),
    ]:
        out = np.zeros(32, dtype=np.float32)
        res = gpusim.run(
            ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
            params={"A": a, "OUT": out, "K": K, "STRIDE": STRIDE,
                    "OUTER_LOOPS": OUTER},
            mode="timing",
        )
        cm = res.cache_metrics
        print(f"  {label}: cycles={res.metrics['cycles']}, "
              f"L1 hit {cm['l1_hit_rate']*100:.1f}%, "
              f"L2 hit {cm['l2_hit_rate']*100:.1f}%")


if __name__ == "__main__":
    main()
