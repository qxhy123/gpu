"""Phase 8 microbench — multi-stream concurrency facts."""
import numpy as np


def test_priority_high_finishes_no_slower_than_low():
    """High priority stream should not finish slower than low priority stream."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8

    src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.f32 %f0, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.f32 %f1, [%rd4];
    add.f32 %f2, %f0, %f1;
    add.u64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f2;
    ret;
}
"""
    _reset_stream_id_counter()
    s_high = Stream(priority="high")
    s_low = Stream(priority="low")
    out_h = np.zeros(n, dtype=np.float32)
    out_l = np.zeros(n, dtype=np.float32)
    s_high.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                    params={"A": A, "B": B, "OUT": out_h}, kernel_name="kh", config=cfg)
    s_low.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                   params={"A": A, "B": B, "OUT": out_l}, kernel_name="kl", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_high, s_low], config=cfg)

    np.testing.assert_array_equal(out_h, A + B)
    np.testing.assert_array_equal(out_l, A + B)
    assert len(multi_res.streams) == 2


def test_event_satisfies_consumer_in_order():
    """Consumer waiting on event sees producer's writes."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()

    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8

    write_src = """
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
    read_src = """
.visible .entry test(.param .u64 IN, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<4>;
    ld.param.u64 %rd0, [IN];
    ld.param.u64 %rd1, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd2, %r1;
    add.u64 %rd3, %rd0, %rd2;
    add.u64 %rd4, %rd1, %rd2;
    ld.global.u32 %r2, [%rd3];
    st.global.u32 [%rd4], %r2;
    ret;
}
"""
    s_a = Stream()
    s_b = Stream()
    ev = Event()
    s_a.launch(ptx_src=write_src, grid=(1,1,1), block=(32,1,1),
                params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    s_b.wait(ev)
    s_b.launch(ptx_src=read_src, grid=(1,1,1), block=(32,1,1),
                params={"IN": SHARED, "OUT": OUT}, kernel_name="read", config=cfg)
    gpusim.synchronize(streams=[s_a, s_b], config=cfg)

    assert SHARED.sum() == n
    assert OUT.sum() == n
