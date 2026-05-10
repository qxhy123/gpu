import numpy as np


def reference(branch: str = "A", n: int = 32):
    return np.ones(n, dtype=np.uint32) if branch == "A" else np.zeros(n, dtype=np.uint32)
