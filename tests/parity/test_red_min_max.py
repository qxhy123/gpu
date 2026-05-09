import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "red_min_max"


def test_red_min_max_correctness():
    """Each thread reads its slot from IN and red.min/red.max into OUT[0..1]."""
    import gpusim
    from gpusim.config.loader import load_default
    rng = np.random.RandomState(0)
    n = 256
    in_arr = rng.randint(0, 1000, size=n).astype(np.int32)
    out = np.zeros(2, dtype=np.int32)
    out[0] = 0x7FFFFFFF   # min seed
    out[1] = -0x80000000  # max seed
    ptx = (_DIR / "kernel.ptx").read_text()
    cfg = load_default()
    res = gpusim.run(
        ptx_src=ptx, grid=(8, 1, 1), block=(32, 1, 1),
        params={"IN": in_arr.copy(), "OUT": out}, mode="timing", config=cfg,
    )
    assert int(out[0]) == int(in_arr.min())
    assert int(out[1]) == int(in_arr.max())
