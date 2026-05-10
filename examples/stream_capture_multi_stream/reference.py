import numpy as np


def reference(n: int = 32, kernels: int = 3):
    return np.full(n, kernels, dtype=np.uint32)
