import numpy as np


def reference(A, B):
    return A.astype(np.float32) @ B.astype(np.float32)
