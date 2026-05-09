def test_nvlink_fabric_default_topology_4_gpus():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    fabric = NvlinkFabric.from_config(cfg, n_gpus=4)
    # all-to-all: 4*3 = 12 links
    assert len(fabric.links) == 12
    for src in range(4):
        for dst in range(4):
            if src != dst:
                assert (src, dst) in fabric.links


def test_nvlink_link_default_bandwidth():
    from gpusim.comm.nvlink import NvlinkLink
    link = NvlinkLink(src_gpu=0, dst_gpu=1,
                        bandwidth_gbps=900.0, latency_cycles=100)
    assert link.bandwidth_gbps == 900.0
    assert link.latency_cycles == 100
    assert link.busy_until == 0


def test_nvlink_transfer_completion_cycle():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    fabric = NvlinkFabric.from_config(cfg, n_gpus=2)
    # 1024 bytes at 900 bytes/cycle ≈ 1 cycle + 100 latency
    completion = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=1024,
                                    arrival_cycle=0)
    assert completion >= 100   # at least latency
    assert completion <= 200


def test_nvlink_transfer_serialization_same_link():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 2
    fabric = NvlinkFabric.from_config(cfg, n_gpus=2)
    c1 = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=900,
                            arrival_cycle=0)
    c2 = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=900,
                            arrival_cycle=0)
    assert c2 > c1


def test_nvlink_transfer_different_links_parallel():
    from gpusim.comm.nvlink import NvlinkFabric
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    fabric = NvlinkFabric.from_config(cfg, n_gpus=4)
    c1 = fabric.transfer(src_gpu=0, dst_gpu=1, n_bytes=900,
                            arrival_cycle=0)
    c2 = fabric.transfer(src_gpu=2, dst_gpu=3, n_bytes=900,
                            arrival_cycle=0)
    assert c1 == c2
