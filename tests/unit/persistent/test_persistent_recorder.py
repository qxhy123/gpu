def test_persistent_kernel_records_is_persistent():
    """Recorder receives KernelLaunch events with is_persistent=True for each item."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.trace.recorder import Recorder
    from gpusim.config.loader import load_default
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
    queue = WorkQueue()
    for _ in range(3):
        queue.push({"OUT": np.zeros(32, dtype=np.uint32)})
    queue.stop()

    rec = Recorder()
    pk = PersistentKernel(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                          params_template={}, work_queue=queue,
                          kernel_name="server")
    pk.start(cfg, recorder=rec)

    # 3 KernelLaunch events recorded with is_persistent=True
    persistent = [e for e in rec.kernel_launch_events if e.is_persistent]
    assert len(persistent) == 3
    assert all(e.parent_kernel_id == -1 for e in persistent)
