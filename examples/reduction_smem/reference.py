# examples/reduction_smem/reference.py
import numpy as np
def reference(a: np.ndarray) -> np.ndarray:
    return np.array([a.sum()], dtype=np.int32)
