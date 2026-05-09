def test_multi_gpu_system_single_gpu_default():
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    sys = MultiGpuSystem.from_config(cfg)
    assert len(sys.gpus) == 1
    assert sys.gpus[0].gpu_id == 0


def test_multi_gpu_system_4_gpus():
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    assert len(sys.gpus) == 4
    assert [g.gpu_id for g in sys.gpus] == [0, 1, 2, 3]
