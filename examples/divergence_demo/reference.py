# examples/divergence_demo/reference.py
import numpy as np
def reference():
    return np.array([100 if i < 16 else 200 for i in range(32)], dtype=np.uint32)
