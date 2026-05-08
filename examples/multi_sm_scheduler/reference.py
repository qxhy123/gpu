import numpy as np


def reference(base: np.ndarray, n_cta: int = 16, ntid: int = 32) -> np.ndarray:
    out = np.zeros_like(base)
    for cta in range(n_cta):
        for t in range(ntid):
            out[cta * ntid + t] = base[cta * ntid + t] + float(cta)
    return out
