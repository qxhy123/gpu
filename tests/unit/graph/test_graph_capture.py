"""Tests for Stream.begin_capture / end_capture (Phase 11 T10/T11)."""
from __future__ import annotations


def test_stream_begin_capture_creates_graph():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    assert s._captured_graph is not None


def test_stream_capture_records_kernel_nodes():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.launch(ptx_src="y", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1")
    g = s.end_capture()
    assert len(g.nodes) == 2
    assert g.nodes[0].kernel_args.kernel_name == "k0"
    assert g.nodes[1].kernel_args.kernel_name == "k1"


def test_stream_capture_implicit_dependency():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.launch(ptx_src="y", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1")
    g = s.end_capture()
    assert len(g.edges) == 1
    assert g.edges[0] == (0, 1)


def test_stream_normal_launch_after_end_capture():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src="x", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0")
    s.end_capture()
    s.launch(ptx_src="y", grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k_normal")
    assert len(s.pending) == 1


def test_capture_then_instantiate_then_launch():
    """Capture 2 launches -> instantiate -> launch -> outputs correct."""
    import numpy as np
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    cfg = load_default()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)

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
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT}, kernel_name="vec_add_0",
              config=cfg)
    s.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": OUT}, kernel_name="vec_add_1",
              config=cfg)
    g = s.end_capture()

    exec = g.instantiate(cfg)
    cycles = exec.launch()

    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
