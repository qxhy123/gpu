"""Phase 14 microbench — persistent + dynamic parallelism facts."""


def test_work_queue_fifo_order():
    from gpusim.persistent.queue import WorkQueue
    q = WorkQueue()
    for i in range(5):
        q.push(i)
    out = []
    while not q.is_empty():
        out.append(q.pop())
    assert out == [0, 1, 2, 3, 4]


def test_persistent_processes_n_items():
    """N items in queue → N iterations of persistent kernel."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
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
    for _ in range(7):
        queue.push({"OUT": np.zeros(32, dtype=np.uint32)})
    queue.stop()

    pk = PersistentKernel(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                          params_template={}, work_queue=queue, kernel_name="t")
    results = pk.start(cfg)
    assert len(results) == 7


def test_dynamic_parallelism_depth_2():
    """Parent → child chain produces depth >= 1."""
    from gpusim.analysis.metrics import dynamic_parallelism_depth
    import pandas as pd
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
        {"stream_id": 1, "parent_kernel_id": 0},
    ])
    assert dynamic_parallelism_depth(df) >= 1
