def test_allgather_correctness():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(8, 1.0, dtype=np.float32)
    recv = np.zeros(32, dtype=np.float32)
    cycles = comm.allgather(send, recv)
    assert cycles > 0
    assert (recv == 1.0).all()
