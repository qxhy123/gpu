def test_stream_construction_assigns_unique_id():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s1 = Stream()
    s2 = Stream()
    assert s1.stream_id != s2.stream_id
    assert s2.stream_id == s1.stream_id + 1


def test_stream_launch_appends_to_pending():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    assert s.is_idle()
    s.launch(ptx_src="entry", grid=(1,1,1), block=(32,1,1),
             params={}, kernel_name="k1")
    assert not s.is_idle()
    assert len(s.pending) == 1
    assert s.pending[0].kernel_name == "k1"


def test_stream_launches_in_order():
    from gpusim.api import Stream, _reset_stream_id_counter
    _reset_stream_id_counter()
    s = Stream()
    for i in range(3):
        s.launch(ptx_src=f"e{i}", grid=(1,1,1), block=(32,1,1),
                 params={}, kernel_name=f"k{i}")
    assert [g.kernel_name for g in s.pending] == ["k0", "k1", "k2"]


def test_result_has_stream_id_default_zero():
    """Single-kernel run via gpusim.run should produce Result with stream_id=0."""
    import gpusim
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    res = gpusim.run(ptx_src=src, grid=(1,1,1), block=(32,1,1),
                      params={}, mode="timing")
    assert res.stream_id == 0


def test_synchronize_drains_two_streams():
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()
    src = """
.entry test() {
    .reg .u32 %r0;
    mov.u32 %r0, %tid.x;
    ret;
}
"""
    cfg = load_default()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k0",
              config=cfg)
    s1.launch(ptx_src=src, grid=(1,1,1), block=(32,1,1), params={}, kernel_name="k1",
              config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    assert 0 in multi_res.streams and 1 in multi_res.streams
    assert len(multi_res.streams[0]) == 1
    assert len(multi_res.streams[1]) == 1
    assert multi_res.streams[0][0].stream_id == 0
    assert multi_res.streams[1][0].stream_id == 1
