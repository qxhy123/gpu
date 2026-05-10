import numpy as np


def reference(n: int = 32, replays: int = 5, kernels: int = 3):
    return np.full(n, replays * kernels, dtype=np.uint32)
