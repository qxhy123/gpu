import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "wgmma_async_pipeline"


def test_wgmma_async_pipeline_matches_numpy():
    import gpusim
    rng = np.random.RandomState(0)
    A = rng.randn(64, 256).astype(np.float16)
    B = rng.randn(256, 128).astype(np.float16)
    out = np.zeros(64 * 128, dtype=np.float32)
    ptx = (_DIR / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(1,1,1), block=(128,1,1),
        params={"A": A.flatten().copy(), "B": B.flatten().copy(),
                "OUT": out, "K_TILES": 16},
        mode="functional",
    )
    expected = (A.astype(np.float32) @ B.astype(np.float32))
    out_2d = out.reshape(64, 128)
    assert np.allclose(out_2d, expected, atol=2e-2), \
        f"max diff = {np.max(np.abs(out_2d - expected))}"
