def test_send_returns_cycles():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=2, system=sys)
    buf = np.arange(64, dtype=np.float32)
    cycles = comm.send(buf, dst_rank=1)
    assert cycles > 0


def test_recv_returns_zero_in_simulator():
    """In simulator, recv is no-op (sender's transfer accounts for it)."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=1, world_size=2, system=sys)
    buf = np.zeros(64, dtype=np.float32)
    cycles = comm.recv(buf, src_rank=0)
    assert cycles == 0


def test_send_records_nvlink_transfer():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=2, system=sys)
    comm._recorder = rec
    buf = np.arange(64, dtype=np.float32)
    comm.send(buf, dst_rank=1)
    assert len(rec.nvlink_transfer_events) == 1
    assert rec.nvlink_transfer_events[0].op_name == "send"
