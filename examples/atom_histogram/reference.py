import numpy as np


def reference(n_threads_per_cta: int = 32, n_cta: int = 8,
              n_bins: int = 16) -> np.ndarray:
    out = np.zeros(n_bins, dtype=np.uint32)
    for cta in range(n_cta):
        for tid in range(n_threads_per_cta):
            out[tid % n_bins] += 1
    return out
