import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    out = np.zeros(2, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    print(f"atom_cas_spinlock: cycles={res.metrics['cycles']}, counter={out[0]}")


if __name__ == "__main__":
    main()
