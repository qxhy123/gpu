import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "atom_histogram"


def test_atom_histogram_correctness():
    """Each thread atomic.add 1 to a bin determined by tid % n_bins."""
    import gpusim
    from gpusim.config.loader import load_default
    n_bins = 16
    n_threads_per_cta = 32
    n_cta = 8
    out = np.zeros(n_bins, dtype=np.uint32)
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(n_cta, 1, 1), block=(n_threads_per_cta, 1, 1),
        params={"OUT": out, "N_BINS": n_bins}, mode="timing", config=cfg,
    )
    expected = np.zeros(n_bins, dtype=np.uint32)
    for cta in range(n_cta):
        for tid in range(n_threads_per_cta):
            expected[tid % n_bins] += 1
    assert (out == expected).all()
