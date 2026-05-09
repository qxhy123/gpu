import numpy as np


def reference(send: np.ndarray, world_size: int) -> np.ndarray:
    return send * world_size
