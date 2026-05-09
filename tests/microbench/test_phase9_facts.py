"""Phase 9 microbench — per-cycle scheduler + Event.elapsed_time facts."""
import numpy as np


def test_event_elapsed_time_returns_positive_int():
    """Event.elapsed_time returns int cycles between two signaled events."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()

    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    s = Stream()
    ev_start = Event(); ev_end = Event()
    s.record(ev_start)
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k", config=cfg)
    s.record(ev_end)
    gpusim.synchronize(streams=[s], config=cfg)

    elapsed = Event.elapsed_time(ev_start, ev_end)
    assert isinstance(elapsed, int)
    assert elapsed >= 0


def test_wait_all_satisfies_consumer_after_all_producers():
    """Stream.wait_all blocks consumer until ALL events signal."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()

    n = 32
    A = np.zeros(n, dtype=np.uint32)
    B = np.zeros(n, dtype=np.uint32)
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
    combine_src = """
.visible .entry test(.param .u64 A, .param .u64 B, .param .u64 OUT) {
    .reg .u64 %rd<6>;
    .reg .u32 %r<6>;
    ld.param.u64 %rd0, [A];
    ld.param.u64 %rd1, [B];
    ld.param.u64 %rd2, [OUT];
    mov.u32 %r0, %tid.x;
    shl.b32 %r1, %r0, 2;
    cvt.u64.u32 %rd3, %r1;
    add.u64 %rd4, %rd0, %rd3;
    ld.global.u32 %r2, [%rd4];
    add.u64 %rd4, %rd1, %rd3;
    ld.global.u32 %r3, [%rd4];
    add.s32 %r4, %r2, %r3;
    add.u64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r4;
    ret;
}
"""
    s_a = Stream(); s_b = Stream(); s_c = Stream()
    ev_a = Event(); ev_b = Event()
    s_a.launch(ptx_src=write_src, grid=(1,1,1), block=(32,1,1),
                params={"OUT": A}, kernel_name="wa", config=cfg)
    s_a.record(ev_a)
    s_b.launch(ptx_src=write_src, grid=(1,1,1), block=(32,1,1),
                params={"OUT": B}, kernel_name="wb", config=cfg)
    s_b.record(ev_b)
    s_c.wait_all([ev_a, ev_b])
    s_c.launch(ptx_src=combine_src, grid=(1,1,1), block=(32,1,1),
                params={"A": A, "B": B, "OUT": OUT}, kernel_name="combine", config=cfg)
    gpusim.synchronize(streams=[s_a, s_b, s_c], config=cfg)

    assert A.sum() == n
    assert B.sum() == n
    assert OUT.sum() == 2 * n
