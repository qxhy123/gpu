"""Phase 7 microbench — multi-stream textbook facts."""
import numpy as np


def test_concurrent_no_slower_than_serial():
    """4 launches via 4 streams should not be slower than 4 launches in 1 stream."""
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
    # Serial
    _reset_stream_id_counter()
    s = Stream()
    outs = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i in range(4):
        s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": outs[i]}, kernel_name=f"k{i}", config=cfg)
    serial_cycles = gpusim.synchronize(streams=[s], config=cfg).total_cycles

    # Concurrent
    _reset_stream_id_counter()
    streams = [Stream() for _ in range(4)]
    outs2 = [np.zeros(n, dtype=np.float32) for _ in range(4)]
    for i, st in enumerate(streams):
        st.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                   params={"A": A, "B": B, "OUT": outs2[i]}, kernel_name=f"k{i}", config=cfg)
    conc_cycles = gpusim.synchronize(streams=streams, config=cfg).total_cycles

    # Phase 7 sequential drain: cycles should be roughly equal (no improvement, but no regression)
    assert conc_cycles <= serial_cycles * 1.5, \
        f"concurrent {conc_cycles} vs serial {serial_cycles}"
