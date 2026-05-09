import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "atom_reduction_smem"


def test_atom_reduction_smem_correctness():
    """N threads atomic.add 1 to a single smem counter; assert counter == N."""
    import gpusim
    from gpusim.config.loader import load_default

    out = np.zeros(1, dtype=np.uint32)
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(1, 1, 1), block=(128, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    assert out[0] == 128
    assert 0 < res.metrics["cycles"] < 50000
