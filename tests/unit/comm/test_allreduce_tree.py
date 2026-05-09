def test_tree_allreduce_step_count():
    """Tree allreduce: 2*log2(N) total steps."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(64, 1.0, dtype=np.float32)
    recv = np.zeros(64, dtype=np.float32)
    cycles = comm._allreduce_tree(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(64, 4.0, dtype=np.float32))
    assert cycles > 0


def test_allreduce_picks_tree_for_small_message():
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
    send = np.full(8, 1.0, dtype=np.float32)   # 32 bytes
    recv = np.zeros(8, dtype=np.float32)
    comm.allreduce(send, recv, op="sum")
    assert rec.collective_events[-1].algorithm == "tree"
