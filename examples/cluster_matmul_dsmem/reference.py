import numpy as np


def reference(A: np.ndarray) -> np.ndarray:
    """Reference: passthrough — each element of OUT equals the corresponding element of A."""
    return A.copy()
