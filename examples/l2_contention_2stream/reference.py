import numpy as np


def reference(n: int = 64) -> np.ndarray:
    out = np.zeros(n, dtype=np.uint32)
    out[0:32] = 1
    out[32:64] = 1
    return out
