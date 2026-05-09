import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    rng = np.random.RandomState(0)
    in_arr = rng.randint(0, 1000, size=256).astype(np.int32)
    out = np.zeros(2, dtype=np.int32)
    out[0] = 0x7FFFFFFF; out[1] = -0x80000000
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(8,1,1), block=(32,1,1),
        params={"IN": in_arr.copy(), "OUT": out}, mode="timing", config=cfg,
    )
    print(f"red_min_max: cycles={res.metrics['cycles']}")
    print(f"  min = {out[0]}, max = {out[1]}")
    print(f"  numpy: min = {int(in_arr.min())}, max = {int(in_arr.max())}")


if __name__ == "__main__":
    main()
