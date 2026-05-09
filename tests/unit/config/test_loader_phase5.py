def test_device_config_cluster_size_default():
    from gpusim.config.schema import DeviceConfig
    cfg = DeviceConfig()
    assert cfg.cluster_size == 1


def test_loader_reads_cluster_size_from_yaml():
    import tempfile
    from pathlib import Path
    yaml_text = """
device:
  n_sm: 8
  cluster_size: 4
  scheduler:
    cta_policy: rr
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_text); path = f.name
    from gpusim.config.loader import load_yaml
    cfg = load_yaml(path)
    assert cfg.cluster_size == 4


def test_loader_default_cluster_size_is_1():
    from gpusim.config.loader import load_default
    cfg = load_default()
    assert cfg.cluster_size == 1
