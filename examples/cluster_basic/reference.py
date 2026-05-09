import numpy as np


def reference(n_cta: int = 2) -> np.ndarray:
    return np.arange(n_cta, dtype=np.uint32)
