import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_sm_scheduler"


def _run(policy: str, n_sm: int = 8):
    import gpusim
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_sm = n_sm
    cfg.scheduler.cta_policy = policy
    rng = np.random.RandomState(0)
    n_cta = 16
    base = (rng.rand(n_cta * 32) * 100).astype(np.float32)
    out = np.zeros(n_cta * 32, dtype=np.float32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(n_cta, 1, 1), block=(32, 1, 1),
        params={"BASE": base.copy(), "OUT": out},
        mode="timing", config=cfg,
    )
    return res, out, base


def test_correctness_rr():
    res, out, base = _run("rr")
    expected = np.zeros(16 * 32, dtype=np.float32)
    for i in range(16):
        for t in range(32):
            expected[i * 32 + t] = base[i * 32 + t] + float(i)
    assert np.allclose(out, expected, atol=1e-5)


def test_correctness_greedy():
    res, out, base = _run("greedy")
    expected = np.zeros(16 * 32, dtype=np.float32)
    for i in range(16):
        for t in range(32):
            expected[i * 32 + t] = base[i * 32 + t] + float(i)
    assert np.allclose(out, expected, atol=1e-5)


def test_greedy_at_least_as_fast_as_rr_on_irregular():
    res_rr, _, _ = _run("rr")
    res_greedy, _, _ = _run("greedy")
    # greedy <= rr (with 5% slack)
    assert res_greedy.metrics["cycles"] <= res_rr.metrics["cycles"] * 1.05, \
        f"greedy={res_greedy.metrics['cycles']} rr={res_rr.metrics['cycles']}"
