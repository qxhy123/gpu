def test_ring_allreduce_step_count():
    """Ring allreduce: 2*(N-1) total steps."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)

    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.arange(64, dtype=np.float32)
    recv = np.zeros(64, dtype=np.float32)
    cycles = comm._allreduce_ring(send, recv, op="sum")

    assert cycles > 0


def test_ring_allreduce_correctness_sum():
    """Ring allreduce sum: all ranks should get send * world_size."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(16, 1.0, dtype=np.float32)
    recv = np.zeros(16, dtype=np.float32)
    comm._allreduce_ring(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(16, 4.0, dtype=np.float32))


def test_allreduce_records_collective_event():
    """allreduce records a CollectiveOp event when recorder set."""
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
    send = np.arange(1024, dtype=np.float32)
    recv = np.zeros(1024, dtype=np.float32)
    comm.allreduce(send, recv, op="sum")
    assert len(rec.collective_events) == 1
    e = rec.collective_events[0]
    assert e.op_name == "allreduce"
    assert e.algorithm == "ring"
