import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    out = np.zeros(16, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(8,1,1), block=(32,1,1),
        params={"OUT": out, "N_BINS": 16}, mode="timing", config=cfg,
    )
    print(f"atom_histogram: cycles={res.metrics['cycles']}")
    print(f"  bins = {list(out)}")


if __name__ == "__main__":
    main()
