import numpy as np, pathlib, gpusim


def main():
    rng = np.random.RandomState(0)
    A = rng.randn(64, 256).astype(np.float16)
    B = rng.randn(256, 128).astype(np.float16)
    out = np.zeros(64 * 128, dtype=np.float32)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=(1, 1, 1), block=(128, 1, 1),
        params={"A": A.flatten().copy(), "B": B.flatten().copy(),
                "OUT": out, "K_TILES": 16},
        mode="functional",
    )
    expected = (A.astype(np.float32) @ B.astype(np.float32))
    diff = np.max(np.abs(out.reshape(64, 128) - expected))
    print(f"wgmma_async_pipeline: max diff={diff:.2e}")


if __name__ == "__main__":
    main()
