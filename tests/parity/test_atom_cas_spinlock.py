import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "atom_cas_spinlock"


def test_atom_cas_spinlock_correctness():
    """N threads use atom.cas as critical section to increment a counter.
    Each thread does counter += 1; final == N."""
    import gpusim
    from gpusim.config.loader import load_default
    n_threads_per_cta = 32
    n_cta = 4
    expected = n_threads_per_cta * n_cta
    out = np.zeros(2, dtype=np.uint32)   # [counter, lock]
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(n_cta, 1, 1), block=(n_threads_per_cta, 1, 1),
        params={"OUT": out}, mode="timing", config=cfg,
    )
    assert int(out[0]) == expected
