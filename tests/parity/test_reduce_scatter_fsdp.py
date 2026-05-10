"""Parity test: reduce_scatter_fsdp — 4-GPU FSDP gradient reduction. Phase 12 T2."""
import numpy as np


def test_reduce_scatter_fsdp_correctness():
    """4-GPU FSDP: each rank gets 1/4 of reduced gradients."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    grads = np.full(64, 1.0, dtype=np.float32)
    my_chunk = np.zeros(16, dtype=np.float32)
    cycles = comm.reduce_scatter(grads, my_chunk, op="sum")
    np.testing.assert_array_equal(my_chunk, np.full(16, 4.0, dtype=np.float32))
    assert cycles > 0
