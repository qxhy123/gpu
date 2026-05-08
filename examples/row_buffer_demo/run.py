import numpy as np
import pathlib
import gpusim


def main():
    n = 1 << 20  # 1 MB float array (4 MB)
    a = np.arange(n, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    print("# stride=1 (sequential, row hits dominate):")
    out = np.zeros(32, dtype=np.float32)
    res = gpusim.run(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                     params={"A": a, "OUT": out, "STRIDE": 1}, mode="timing")
    print(f"  cycles={res.metrics['cycles']}")
    print("# stride=131072 (= 512 KB / 4 B per element, row misses dominate):")
    out = np.zeros(32, dtype=np.float32)
    res = gpusim.run(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                     params={"A": a, "OUT": out, "STRIDE": 131072}, mode="timing")
    print(f"  cycles={res.metrics['cycles']}")


if __name__ == "__main__":
    main()
