import numpy as np


def reference(grads, world_size, rank):
    chunk = grads.size // world_size
    return grads[rank*chunk:(rank+1)*chunk] * world_size
