# examples/coalescing_demo/reference.py
import numpy as np
def reference(a, stride):
    return a[: 32*stride : stride].copy()
