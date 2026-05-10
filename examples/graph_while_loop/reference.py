import numpy as np


def reference(n: int = 32, iterations: int = 4):
    return np.full(n, iterations, dtype=np.uint32)
