def test_comm_construction():
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    c = Comm(rank=0, world_size=4, system=sys)
    assert c.rank == 0
    assert c.world_size == 4


def test_comm_rank_validation():
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    with pytest.raises(ValueError, match="rank"):
        Comm(rank=5, world_size=4, system=sys)
    with pytest.raises(ValueError, match="world_size"):
        Comm(rank=0, world_size=8, system=sys)
