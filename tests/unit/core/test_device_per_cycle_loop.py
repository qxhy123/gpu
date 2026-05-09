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


def test_cta_retire_decrements_stream_in_flight_ctas():
    """When SM retires a CTA, stream.in_flight_ctas decreases by 1."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    s = Stream()
    s.launch(ptx_src=src, grid=(2,1,1), block=(32,1,1), params={}, kernel_name="k")
    multi_res = gpusim.synchronize(streams=[s], config=cfg)
    # After synchronize, stream should have in_flight_ctas == 0
    assert s.in_flight_ctas == 0


def test_on_cta_retired_helper_exists():
    """Phase 9 helper for retire tracking."""
    from gpusim.core.device import Device
    from gpusim.config.loader import load_default
    cfg = load_default()
    d = Device(cfg)
    assert hasattr(d, "_on_cta_retired")


def test_on_cta_retired_decrements_stream_counter():
    """Direct call: decrement when stream is in _active_streams."""
    from gpusim.core.device import Device
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    cfg = load_default()
    d = Device(cfg)
    s = Stream()
    s.in_flight_ctas = 5
    d._active_streams = [s]
    d._on_cta_retired(s.stream_id)
    assert s.in_flight_ctas == 4
