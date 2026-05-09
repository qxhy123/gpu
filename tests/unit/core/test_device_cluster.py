def test_device_cluster_size_1_equivalent_to_phase4():
    """cluster_size=1 default → byte-for-byte Phase 4 behavior."""
    import numpy as np
    from gpusim.config.schema import DeviceConfig
    from gpusim.core.device import Device
    from gpusim.frontend.parser import parse
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<3>;
    .reg .u32 %r<3>;
    .reg .pred %p0;
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r0, %ctaid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd2, %rd1, %rd2;
    mov.u32 %r2, %tid.x;
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra END;
    st.global.u32 [%rd2], %r0;
END:
    ret;
}
"""
    cfg = DeviceConfig(n_sm=4, cluster_size=1)
    out = np.zeros(8, dtype=np.uint32)
    dev = Device(cfg)
    res = dev.run(kernel=parse(src, "<test>"), grid=(8,1,1), block=(32,1,1),
                   params={"OUT": out})
    assert (out == np.arange(8, dtype=np.uint32)).all()


def test_device_cluster_size_2_dispatches_pairs():
    """cluster_size=2 dispatches CTAs 0,1 then 2,3 etc. as pairs."""
    import numpy as np
    from gpusim.config.schema import DeviceConfig
    from gpusim.core.device import Device
    from gpusim.frontend.parser import parse
    src = """
.entry test(.param .u64 OUT) {
    .reg .u64 %rd<3>;
    .reg .u32 %r<3>;
    .reg .pred %p0;
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r0, %ctaid.x;
    mul.lo.s32 %r1, %r0, 4;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd2, %rd1, %rd2;
    mov.u32 %r2, %tid.x;
    setp.eq.u32 %p0, %r2, 0;
    @!%p0 bra END;
    st.global.u32 [%rd2], %r0;
END:
    ret;
}
"""
    cfg = DeviceConfig(n_sm=4, cluster_size=2)
    out = np.zeros(4, dtype=np.uint32)
    dev = Device(cfg)
    res = dev.run(kernel=parse(src, "<test>"), grid=(4,1,1), block=(32,1,1),
                   params={"OUT": out})
    assert (out == np.arange(4, dtype=np.uint32)).all()


def test_device_cluster_size_must_divide_grid():
    """grid_size % cluster_size != 0 → ValueError."""
    import pytest
    from gpusim.config.schema import DeviceConfig
    from gpusim.core.device import Device
    from gpusim.frontend.parser import parse
    src = ".entry test() { ret; }"
    cfg = DeviceConfig(n_sm=4, cluster_size=4)
    dev = Device(cfg)
    with pytest.raises(ValueError, match="cluster_size"):
        dev.run(kernel=parse(src, "<t>"), grid=(7,1,1), block=(32,1,1), params={})
