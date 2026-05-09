import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "event_producer_consumer"


def test_event_producer_consumer_correctness():
    """Stream A writes X → record(ev) → Stream B wait(ev) → reads X."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()

    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8

    ptx_write = (_DIR / "kernel_write.ptx").read_text()
    ptx_read = (_DIR / "kernel_read.ptx").read_text()

    s_a = Stream()
    s_b = Stream()
    ev = Event()

    s_a.launch(ptx_src=ptx_write, grid=(1, 1, 1), block=(32, 1, 1),
               params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    s_b.wait(ev)
    s_b.launch(ptx_src=ptx_read, grid=(1, 1, 1), block=(32, 1, 1),
               params={"IN": SHARED, "OUT": OUT}, kernel_name="read", config=cfg)

    multi_res = gpusim.synchronize(streams=[s_a, s_b], config=cfg)

    assert SHARED.sum() == n
    assert OUT.sum() == n
