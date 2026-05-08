import numpy as np


def reference(A, B):
    """Compute A @ B in FP32, matching the FP32-accumulator reference result."""
    return A.astype(np.float32) @ B.astype(np.float32)
