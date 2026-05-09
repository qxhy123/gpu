import numpy as np
import pathlib
import gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.cluster_size = 4
    cfg.n_sm = 4
    rng = np.random.RandomState(0)
    A = (rng.rand(128) * 100).astype(np.float32)
    B = np.zeros(1, dtype=np.float32)
    out = np.zeros(128, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4, 1, 1), block=(128, 1, 1),
        params={"A": A.copy(), "B": B, "OUT": out},
        mode="timing", config=cfg,
    )
    diff = float(np.max(np.abs(out - A)))
    print(
        f"cluster_matmul_dsmem (simplified): "
        f"cycles={res.metrics['cycles']}, max diff={diff:.2e}"
    )


if __name__ == "__main__":
    main()
