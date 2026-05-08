import numpy as np
import pathlib
import gpusim


def main():
    rng = np.random.RandomState(0)
    ro_in = (rng.rand(40000) * 100).astype(np.float32)
    out = np.zeros(8 * 32, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(8, 1, 1), block=(32, 1, 1),
        params={"RO_IN": ro_in, "OUT": out, "RO_LEN": 40000},
        mode="timing",
    )
    print(f"l2_sharing_demo: cycles={res.metrics['cycles']}")
    print(f"  cache_summary: {res.cache_summary()}")


if __name__ == "__main__":
    main()
