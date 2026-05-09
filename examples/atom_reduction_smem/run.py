# examples/atom_reduction_smem/run.py
import numpy as np
import pathlib
import gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    out = np.zeros(1, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(1, 1, 1), block=(128, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    print(f"atom_reduction_smem: cycles={res.metrics['cycles']}, count={out[0]}")


if __name__ == "__main__":
    main()
