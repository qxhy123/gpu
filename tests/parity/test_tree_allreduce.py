def test_tree_allreduce_correctness():
    """4-GPU tree allreduce on a 64-byte buffer (small msg → tree path)."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(16, 1.0, dtype=np.float32)   # 64 bytes
    recv = np.zeros(16, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(16, 4.0, dtype=np.float32))
    assert rec.collective_events[-1].algorithm == "tree"
    assert cycles > 0
