def test_persistent_kernel_processes_all_items():
    """PersistentKernel.start() processes all queued items + stops on queue stop."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    cfg = load_default()

    src = """
.visible .entry test(.param .u64 OUT) {
    .reg .u64 %rd<4>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd1, %r1;
    add.u64 %rd2, %rd0, %rd1;
    mov.u32 %r2, 1;
    st.global.u32 [%rd2], %r2;
    ret;
}
"""
    queue = WorkQueue()
    out_buffers = [np.zeros(32, dtype=np.uint32) for _ in range(3)]
    for ob in out_buffers:
        queue.push({"OUT": ob})
    queue.stop()

    pk = PersistentKernel(
        ptx_src=src, grid=(1,1,1), block=(32,1,1),
        params_template={}, work_queue=queue, kernel_name="persistent_k",
    )
    results = pk.start(cfg)

    assert len(results) == 3
    for ob in out_buffers:
        assert ob.sum() == 32   # each thread wrote 1


def test_persistent_kernel_stops_on_empty_queue():
    """PersistentKernel exits when queue is empty + stopped (returns empty list)."""
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default
    cfg = load_default()
    queue = WorkQueue()
    queue.stop()    # empty + stopped
    pk = PersistentKernel(
        ptx_src="x", grid=(1,1,1), block=(32,1,1),
        params_template={}, work_queue=queue,
    )
    results = pk.start(cfg)
    assert results == []
