# examples/tiled_matmul/reference.py
import numpy as np
def reference(A: np.ndarray, B: np.ndarray) -> np.ndarray: return A @ B
