import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cluster_matmul_dsmem"


def test_cluster_matmul_dsmem_correctness():
    import gpusim
    from gpusim.config.loader import load_default

    rng = np.random.RandomState(0)
    cfg = load_default()
    cfg.cluster_size = 4
    cfg.n_sm = 4
    A_data = (rng.rand(128) * 100).astype(np.float32)
    B_unused = np.zeros(1, dtype=np.float32)
    out = np.zeros(128, dtype=np.float32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx,
        grid=(4, 1, 1),
        block=(128, 1, 1),
        params={"A": A_data.copy(), "B": B_unused, "OUT": out},
        mode="timing",
        config=cfg,
    )
    assert np.allclose(out, A_data, atol=0), (
        f"max diff = {float(np.max(np.abs(out - A_data))):.2e}"
    )
