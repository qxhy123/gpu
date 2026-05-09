def test_device_helpers_exist():
    """Phase 9 helpers for per-cycle main loop."""
    from gpusim.core.device import Device
    from gpusim.config.loader import load_default
    cfg = load_default()
    d = Device(cfg)
    assert hasattr(d, "_available_sms")
    assert hasattr(d, "_dispatch_cta_to_sm")
    assert hasattr(d, "_stream_grid_retired")


def test_available_sms_returns_list():
    from gpusim.core.device import Device
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_sm = 4
    d = Device(cfg)
    sms = d._available_sms()
    assert isinstance(sms, list)
    assert len(sms) <= 4
