import numpy as np


def test_device_single_sm_degenerate_runs_phase1_kernel():
    from gpusim.config.schema import DeviceConfig
    from gpusim.core.device import Device
    from gpusim.frontend.parser import parse

    cfg = DeviceConfig(n_sm=1)
    src = """
.visible .entry test(.param .u64 OUT)
{
    .reg .u64 %rd<3>;
    .reg .u32 %r<3>;
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    cvt.u64.u32 %rd2, %r2;
    add.u64 %rd2, %rd1, %rd2;
    st.global.u32 [%rd2], %r1;
    ret;
}
"""
    k = parse(src, "<test>")
    out = np.zeros(32, dtype=np.uint32)
    dev = Device(cfg)
    res = dev.run(kernel=k, grid=(1, 1, 1), block=(32, 1, 1),
                   params={"OUT": out})
    assert res.cycles > 0
    assert list(out) == list(range(32))


def test_device_n_sm_attribute():
    from gpusim.config.schema import DeviceConfig
    from gpusim.core.device import Device
    cfg = DeviceConfig(n_sm=8)
    dev = Device(cfg)
    assert dev.n_sm == 8
