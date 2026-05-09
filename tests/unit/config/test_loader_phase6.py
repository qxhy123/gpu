def test_cache_config_atomic_fields_default():
    from gpusim.config.schema import CacheConfig
    cfg = CacheConfig()
    assert cfg.atomic_op_latency == 10
    assert cfg.atomic_queue_capacity == 32
    assert cfg.smem_atomic_op_extra_latency == 4


def test_loader_reads_atomic_fields_from_yaml():
    import tempfile
    from pathlib import Path
    yaml_text = """
device:
  n_sm: 8
cache:
  l1_size_bytes: 131072
  atomic_op_latency: 15
  atomic_queue_capacity: 64
  smem_atomic_op_extra_latency: 6
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text); path = f.name
    from gpusim.config.loader import load_yaml
    cfg = load_yaml(path)
    assert cfg.cache.atomic_op_latency == 15
    assert cfg.cache.atomic_queue_capacity == 64
    assert cfg.cache.smem_atomic_op_extra_latency == 6
