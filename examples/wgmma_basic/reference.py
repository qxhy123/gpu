"""reference.py — NumPy reference for wgmma_basic (fp16 matmul accumulating to fp32)."""
import numpy as np


def reference_matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute D = A @ B using fp32 accumulation from fp16 inputs.

    Parameters
    ----------
    A : (64, 16) float16
    B : (16, 128) float16

    Returns
    -------
    D : (64, 128) float32
    """
    if A.dtype != np.float16:
        A = A.astype(np.float16)
    if B.dtype != np.float16:
        B = B.astype(np.float16)
    return A.astype(np.float32) @ B.astype(np.float32)


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    A = rng.randn(64, 16).astype(np.float16)
    B = rng.randn(16, 128).astype(np.float16)
    D = reference_matmul(A, B)
    print(f"D shape: {D.shape}, dtype: {D.dtype}")
    print(f"D[:2, :4]:\n{D[:2, :4]}")
