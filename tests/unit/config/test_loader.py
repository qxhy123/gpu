from gpusim.config.loader import load_default, load_yaml
from gpusim.config.schema import DeviceConfig

def test_default_loads():
    c = load_default()
    assert isinstance(c, DeviceConfig)
    assert c.sm.sub_cores == 4
    assert c.sm.warps_per_sm == 64
    assert c.sm.smem_banks == 32
    assert c.sm.regfile.banks == 4
    assert c.sm.scheduler.policy == "gto"

def test_overrides_via_yaml(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("sm:\n  scheduler:\n    policy: lrr\n")
    c = load_yaml(p)
    assert c.sm.scheduler.policy == "lrr"
    assert c.sm.sub_cores == 4

def test_default_loads_cache_section():
    c = load_default()
    assert c.cache.l1_size_bytes == 131072
    assert c.cache.l1_ways == 4
    assert c.cache.l1_line_bytes == 128
    assert c.cache.mshr_slots == 16
    assert c.cache.l1_hit_latency == 25
    assert c.cache.l2_size_bytes == 4 * 1024 * 1024
    assert c.cache.l2_ways == 16
    assert c.cache.l2_hit_latency == 200

def test_default_loads_hbm_section():
    c = load_default()
    assert c.hbm.channels == 8
    assert c.hbm.banks_per_channel == 16
    assert c.hbm.row_size_bytes == 4096
    assert c.hbm.row_hit_latency == 10
    assert c.hbm.row_miss_latency == 30
