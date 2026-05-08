import numpy as np


def reference(a, K, STRIDE):
    return a[(K-1)*STRIDE : (K-1)*STRIDE + 32]
