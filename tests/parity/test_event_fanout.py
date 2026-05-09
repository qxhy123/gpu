import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "event_fanout"


def test_event_fanout_correctness():
    """1 producer event satisfies 3 consumer streams."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()

    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    out_b = np.zeros(n, dtype=np.uint32)
    out_c = np.zeros(n, dtype=np.uint32)
    out_d = np.zeros(n, dtype=np.uint32)

    cfg = load_default()
    cfg.n_sm = 8

    ptx_write = (_DIR / "kernel_write.ptx").read_text()
    ptx_read = (_DIR / "kernel_read.ptx").read_text()

    s_a = Stream()
    s_b = Stream()
    s_c = Stream()
    s_d = Stream()
    ev = Event()

    s_a.launch(ptx_src=ptx_write, grid=(1, 1, 1), block=(32, 1, 1),
               params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)

    for s, out in [(s_b, out_b), (s_c, out_c), (s_d, out_d)]:
        s.wait(ev)
        s.launch(ptx_src=ptx_read, grid=(1, 1, 1), block=(32, 1, 1),
                 params={"IN": SHARED, "OUT": out},
                 kernel_name=f"read_{s.stream_id}", config=cfg)

    multi_res = gpusim.synchronize(streams=[s_a, s_b, s_c, s_d], config=cfg)

    assert SHARED.sum() == n
    assert out_b.sum() == n
    assert out_c.sum() == n
    assert out_d.sum() == n
