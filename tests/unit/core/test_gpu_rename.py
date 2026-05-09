"""Phase 10 T1: GPU alias + gpu_id field."""


def test_device_alias_to_gpu():
    """Phase 10 backward compat: GPU is alias for Device."""
    from gpusim.core.device import Device, GPU
    assert Device is GPU


def test_gpu_has_gpu_id_field():
    from gpusim.core.device import GPU
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = GPU(cfg)
    assert hasattr(g, "gpu_id")


def test_gpu_with_explicit_id():
    from gpusim.core.device import GPU
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = GPU(cfg, gpu_id=2)
    assert g.gpu_id == 2
