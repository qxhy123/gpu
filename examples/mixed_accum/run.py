"""mixed_accum: compare FP16 vs FP32 accumulator precision over 64 K-tiles."""
import pathlib, numpy as np, gpusim

_DIR = pathlib.Path(__file__).parent


def main():
    rng = np.random.RandomState(42)
    A_full = rng.randn(16, 16 * 64).astype(np.float16)
    B_full = rng.randn(16 * 64, 8).astype(np.float16)
    A = A_full.flatten().copy()
    B = B_full.flatten().copy()
    expected = A_full.astype(np.float32) @ B_full.astype(np.float32)

    print("# mixed_accum: FP16 vs FP32 accumulator (64 K-tile iterations)")
    print(f"{'variant':<14} {'cycles':<10} {'max diff vs fp32 ref'}")

    for variant, out_dtype in (("fp32_accum", np.float32), ("fp16_accum", np.float16)):
        out = np.zeros(16 * 8, dtype=out_dtype)
        ptx = (_DIR / f"kernel_{variant}.ptx").read_text()
        res = gpusim.run(
            ptx_src=ptx,
            grid=(1, 1, 1),
            block=(32, 1, 1),
            params={"A": A, "B": B, "OUT": out, "K_ITERS": 64},
            mode="timing",
        )
        diff = np.max(np.abs(out.reshape(16, 8).astype(np.float32) - expected))
        cycles = res.metrics.get("cycles", "?")
        print(f"{variant:<14} {str(cycles):<10} {diff:.4e}")


if __name__ == "__main__":
    main()
