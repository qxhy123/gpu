import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_event_fan_in"


def test_multi_event_fan_in_correctness():
    """2 producers (s_a, s_b) → 1 consumer (s_c) using s_c.wait_all([ev_a, ev_b])."""
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

    ptx_write = (_DIR / "kernel_write.ptx").read_text()
    ptx_combine = (_DIR / "kernel_combine.ptx").read_text()

    s_a = Stream()
    s_b = Stream()
    s_c = Stream()
    ev_a = Event(); ev_b = Event()

    s_a.launch(ptx_src=ptx_write, grid=(1,1,1), block=(32,1,1),
               params={"OUT": A}, kernel_name="write_a", config=cfg)
    s_a.record(ev_a)

    s_b.launch(ptx_src=ptx_write, grid=(1,1,1), block=(32,1,1),
               params={"OUT": B}, kernel_name="write_b", config=cfg)
    s_b.record(ev_b)

    s_c.wait_all([ev_a, ev_b])
    s_c.launch(ptx_src=ptx_combine, grid=(1,1,1), block=(32,1,1),
               params={"A": A, "B": B, "OUT": OUT}, kernel_name="combine", config=cfg)

    multi_res = gpusim.synchronize(streams=[s_a, s_b, s_c], config=cfg)

    assert A.sum() == n
    assert B.sum() == n
    assert OUT.sum() == 2 * n
