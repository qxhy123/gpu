"""Phase 10 T2: Config.n_gpus + NvlinkConfig."""


def test_config_n_gpus_default_1():
    from gpusim.config.loader import load_default
    cfg = load_default()
    assert cfg.n_gpus == 1


def test_config_nvlink_section():
    from gpusim.config.schema import NvlinkConfig
    nv = NvlinkConfig()
    assert nv.bandwidth_gbps == 900.0
    assert nv.latency_cycles == 100
    assert nv.topology == "all_to_all"
    assert nv.half_duplex is True
