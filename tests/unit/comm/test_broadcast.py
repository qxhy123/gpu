def test_broadcast_correctness():
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    buf = np.arange(32, dtype=np.float32)
    cycles = comm.broadcast(buf, root=0)
    assert cycles > 0
