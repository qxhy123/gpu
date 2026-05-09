import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default(); cfg.cluster_size = 4; cfg.n_sm = 4
    rng = np.random.RandomState(0)
    src = (rng.rand(256) * 100).astype(np.float32)
    out = np.zeros(256, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
        params={"SRC": src.copy(), "OUT": out}, mode="timing", config=cfg,
    )
    diff = float(np.max(np.abs(out - src)))
    print(f"cluster_tma_pipeline: cycles={res.metrics['cycles']}, max diff={diff:.2e}")


if __name__ == "__main__":
    main()
