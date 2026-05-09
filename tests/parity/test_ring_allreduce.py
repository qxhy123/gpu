def test_ring_allreduce_correctness():
    """4-GPU ring allreduce on a 1024-byte buffer."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)

    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(256, 1.0, dtype=np.float32)
    recv = np.zeros(256, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")

    np.testing.assert_array_equal(recv, np.full(256, 4.0, dtype=np.float32))
    assert cycles > 0
