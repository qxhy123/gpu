"""run.py — Driver for the wgmma_basic example.

Runs the Hopper wgmma m64n128k16 kernel via gpusim and compares against
the NumPy reference.
"""
import pathlib
import numpy as np
import gpusim
from reference import reference_matmul

_DIR = pathlib.Path(__file__).resolve().parent


def run(seed: int = 0, verbose: bool = True) -> bool:
    """Run the kernel and verify against reference. Returns True on success."""
    rng = np.random.RandomState(seed)
    A = rng.randn(64, 16).astype(np.float16)
    B = rng.randn(16, 128).astype(np.float16)
    out = np.zeros(64 * 128, dtype=np.float32)

    ptx = (_DIR / "kernel.ptx").read_text()

    res = gpusim.run(
        ptx_src=ptx,
        grid=(1, 1, 1),
        block=(128, 1, 1),
        params={"A": A.flatten().copy(), "B": B.flatten().copy(), "OUT": out},
        mode="functional",
    )

    expected = reference_matmul(A, B)
    out_2d = out.reshape(64, 128)
    max_diff = float(np.max(np.abs(out_2d - expected)))
    ok = np.allclose(out_2d, expected, atol=1e-2)

    if verbose:
        print(f"wgmma_basic: A={A.shape} B={B.shape} -> D={out_2d.shape}")
        print(f"  max |diff| = {max_diff:.6f}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  out_2d[:2,:4] = {out_2d[:2,:4]}")
            print(f"  expected[:2,:4] = {expected[:2,:4]}")

    return ok


if __name__ == "__main__":
    import sys
    ok = run(verbose=True)
    sys.exit(0 if ok else 1)
