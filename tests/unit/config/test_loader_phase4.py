def test_cta_scheduler_config_default():
    from gpusim.config.schema import CtaSchedulerConfig
    cfg = CtaSchedulerConfig()
    assert cfg.cta_policy == "rr"


def test_device_config_default():
    from gpusim.config.schema import DeviceConfig
    cfg = DeviceConfig()
    assert cfg.n_sm == 8
    assert cfg.scheduler.cta_policy == "rr"
    assert cfg.cache.l2_size_bytes == 4 * 1024 * 1024
    assert cfg.hbm.channels == 8
    assert cfg.sm.sub_cores == 4


def test_cache_config_has_l2_mshr_slots():
    from gpusim.config.schema import CacheConfig
    cfg = CacheConfig()
    assert cfg.l2_mshr_slots == 32


def test_tensor_core_config_has_bulk_store_fields():
    from gpusim.config.schema import TensorCoreConfig
    cfg = TensorCoreConfig()
    assert cfg.bulk_store_queue_capacity == 16
    assert cfg.bulk_store_latency_per_line == 4


import tempfile
from pathlib import Path


def test_loader_legacy_yaml_falls_back_to_single_sm():
    """Legacy yaml without `device:` node merges with device-first default.
    After T3, the default has device: {n_sm: 8}, so merged result inherits n_sm=8.
    Cache and HBM overrides from legacy yaml are applied via dict merge."""
    yaml_text = """
cache:
  l1_size_bytes: 131072
  l2_size_bytes: 4194304
  l2_mshr_slots: 32
hbm:
  channels: 8
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    from gpusim.config.loader import load_yaml
    cfg = load_yaml(path)
    from gpusim.config.schema import DeviceConfig
    assert isinstance(cfg, DeviceConfig)
    assert cfg.cache.l2_size_bytes == 4194304
    assert cfg.hbm.channels == 8
    assert cfg.sm.sub_cores == 4


def test_loader_device_first_yaml():
    yaml_text = """
device:
  n_sm: 4
  scheduler:
    cta_policy: greedy

sm:
  sub_cores: 4
  warps_per_sm: 64
  threads_per_sm: 2048
  max_ctas_per_sm: 32
  regs_per_sm: 65536
  smem_per_sm_bytes: 49152
  smem_banks: 32

cache:
  l1_size_bytes: 131072
  l2_size_bytes: 4194304
  l2_mshr_slots: 64

hbm:
  channels: 8
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text)
        path = f.name
    from gpusim.config.loader import load_yaml
    cfg = load_yaml(path)
    assert cfg.n_sm == 4
    assert cfg.scheduler.cta_policy == "greedy"
    assert cfg.cache.l2_mshr_slots == 64


def test_load_default_uses_device_first():
    from gpusim.config.loader import load_default
    cfg = load_default()
    # default_hopper.yaml will be restructured in T3 to have n_sm: 8
    # For T2, default still has legacy form → loader falls back to n_sm=1
    # T3 will set n_sm to 8 by restructuring yaml
    # For now, just check it returns DeviceConfig
    from gpusim.config.schema import DeviceConfig
    assert isinstance(cfg, DeviceConfig)


def test_default_yaml_has_device_node():
    from gpusim.config.loader import load_default
    cfg = load_default()
    from gpusim.config.schema import DeviceConfig
    assert isinstance(cfg, DeviceConfig)
    assert cfg.n_sm == 8
    assert cfg.scheduler.cta_policy == "rr"
    assert cfg.cache.l2_mshr_slots == 32
    assert cfg.sm.tensor_core.bulk_store_queue_capacity == 16
