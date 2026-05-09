import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cluster_tma_pipeline"


def test_cluster_tma_pipeline_correctness():
    import gpusim
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.cluster_size = 4; cfg.n_sm = 4
    rng = np.random.RandomState(0)
    src_arr = (rng.rand(256) * 100).astype(np.float32)
    out = np.zeros(256, dtype=np.float32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4, 1, 1), block=(32, 1, 1),
        params={"SRC": src_arr.copy(), "OUT": out},
        mode="functional", config=cfg,
    )
    assert np.allclose(out, src_arr, atol=1e-5)
