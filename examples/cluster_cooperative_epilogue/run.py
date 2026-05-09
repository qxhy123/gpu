import numpy as np, pathlib, gpusim
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.cluster_size = 4; cfg.n_sm = 4
    out = np.zeros(128, dtype=np.uint32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4,1,1), block=(32,1,1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    print(f"cluster_cooperative_epilogue: cycles={res.metrics['cycles']}")
    print(f"  out[0:4] = {list(out[0:4])}, out[32:36] = {list(out[32:36])}")


if __name__ == "__main__":
    main()
