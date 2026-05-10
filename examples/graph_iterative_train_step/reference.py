import numpy as np


def reference(weights, grads, n_epochs):
    return weights - grads * n_epochs
