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
