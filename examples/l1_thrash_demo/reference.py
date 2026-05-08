import numpy as np


def reference(a, K, STRIDE, OUTER_LOOPS=8):
    """Last value seen by each thread: a[(K-1)*STRIDE + tid]."""
    return a[(K - 1) * STRIDE : (K - 1) * STRIDE + 32]
