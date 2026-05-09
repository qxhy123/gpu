import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cluster_basic"


def test_cluster_basic_correctness():
    import gpusim
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.cluster_size = 2
    out = np.zeros(2, dtype=np.uint32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(2, 1, 1), block=(32, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    assert (out == np.array([0, 1], dtype=np.uint32)).all()
    assert 0 < res.metrics["cycles"] < 5000
