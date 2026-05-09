def reference(n_threads_per_cta: int = 32, n_cta: int = 4) -> int:
    return n_threads_per_cta * n_cta
