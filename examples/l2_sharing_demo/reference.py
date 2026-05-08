import numpy as np


def reference(ro_in: np.ndarray, n_cta: int = 8) -> np.ndarray:
    out = np.zeros(n_cta * 32, dtype=np.float32)
    for cta in range(n_cta):
        for t in range(32):
            ridx = cta * 8 + t
            out[cta * 32 + t] = ro_in[ridx]
    return out
