# examples/atom_reduction_smem/reference.py


def reference(n_threads: int) -> int:
    """Reference: N threads each add 1 to a counter → result is N."""
    return n_threads
