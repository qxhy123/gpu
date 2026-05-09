import numpy as np


def reference(in_arr: np.ndarray) -> tuple[int, int]:
    return int(in_arr.min()), int(in_arr.max())
