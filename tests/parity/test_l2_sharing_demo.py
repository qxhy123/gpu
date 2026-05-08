import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "l2_sharing_demo"


def _run():
    import gpusim
    rng = np.random.RandomState(0)
    n_cta = 8
    n_per_cta = 32
    ro_in = (rng.rand(40000) * 100).astype(np.float32)
    out = np.zeros(n_cta * n_per_cta, dtype=np.float32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(n_cta, 1, 1), block=(32, 1, 1),
        params={"RO_IN": ro_in.copy(), "OUT": out, "RO_LEN": 40000},
        mode="timing",
    )
    return res, out, ro_in


def test_correctness():
    res, out, ro_in = _run()
    assert res.metrics["cycles"] > 0
    assert (out != 0).any()


def test_no_runaway():
    res, _, _ = _run()
    assert res.metrics["cycles"] < 5_000_000
