import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "cluster_cooperative_epilogue"


def test_cluster_cooperative_epilogue_correctness():
    """4-CTA cluster: each CTA fills its smem with rank-tagged data; CTA 0
    uses cluster TMA store to write all 4 CTAs' data to OUT (each CTA's slice
    placed at rank * 32 in OUT)."""
    import gpusim
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.cluster_size = 4; cfg.n_sm = 4
    out = np.zeros(128, dtype=np.uint32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(4, 1, 1), block=(32, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    expected = np.zeros(128, dtype=np.uint32)
    for r in range(4):
        for i in range(32):
            expected[r * 32 + i] = r * 1000 + i
    assert (out == expected).all()
