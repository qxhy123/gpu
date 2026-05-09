import numpy as np


def reference(n_cta: int = 4, n_per_cta: int = 32) -> np.ndarray:
    out = np.zeros(n_cta * n_per_cta, dtype=np.uint32)
    for r in range(n_cta):
        for i in range(n_per_cta):
            out[r * n_per_cta + i] = r * 1000 + i
    return out
