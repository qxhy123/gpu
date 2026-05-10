"""Unit tests for Phase 14 dynamic parallelism — device_launch + drain."""


def test_device_launch_appends_to_pending():
    from gpusim.persistent.dynamic import (
        device_launch, _pending_child_launches, reset_pending_child_launches,
    )
    reset_pending_child_launches()
    device_launch(parent_kernel_id=0, ptx_src="x",
                  grid=(1, 1, 1), block=(32, 1, 1),
                  params={}, kernel_name="child")
    assert len(_pending_child_launches) == 1
    assert _pending_child_launches[0]["parent_kernel_id"] == 0


def test_drain_pending_child_launches_processes_all():
    """drain processes pending launches and returns Results."""
    import numpy as np
    from gpusim.persistent.dynamic import (
        device_launch, drain_pending_child_launches, reset_pending_child_launches,
    )
    from gpusim.config.loader import load_default
    reset_pending_child_launches()

    cfg = load_default()
    src = """
.visible .entry test(.param .u64 OUT) {
    .reg .u64 %rd<4>; .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 7; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    out = np.zeros(32, dtype=np.uint32)
    device_launch(parent_kernel_id=0, ptx_src=src,
                  grid=(1, 1, 1), block=(32, 1, 1),
                  params={"OUT": out}, kernel_name="child")

    results = drain_pending_child_launches(cfg)
    assert len(results) == 1
    assert out.sum() == 32 * 7   # each thread wrote 7


def test_reset_pending_clears_state():
    from gpusim.persistent.dynamic import (
        device_launch, _pending_child_launches, reset_pending_child_launches,
    )
    reset_pending_child_launches()
    device_launch(parent_kernel_id=0, ptx_src="x",
                  grid=(1, 1, 1), block=(32, 1, 1),
                  params={}, kernel_name="child")
    assert len(_pending_child_launches) == 1
    reset_pending_child_launches()
    assert len(_pending_child_launches) == 0


def test_drain_records_parent_kernel_id():
    """drain records KernelLaunch events with correct parent_kernel_id."""
    import numpy as np
    from gpusim.persistent.dynamic import (
        device_launch, drain_pending_child_launches, reset_pending_child_launches,
    )
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    reset_pending_child_launches()

    cfg = load_default()
    src = """
.visible .entry test(.param .u64 OUT) {
    .reg .u64 %rd<4>; .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x; shl.b32 %r1, %r0, 2; cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1; st.global.u32 [%rd2], %r2;
    ret;
}
"""
    out = np.zeros(32, dtype=np.uint32)
    device_launch(parent_kernel_id=42, ptx_src=src,
                  grid=(1, 1, 1), block=(32, 1, 1),
                  params={"OUT": out}, kernel_name="child_k")

    rec = Recorder()
    drain_pending_child_launches(cfg, recorder=rec)

    assert len(rec.kernel_launch_events) == 1
    evt = rec.kernel_launch_events[0]
    assert evt.parent_kernel_id == 42
    assert evt.is_persistent is False
