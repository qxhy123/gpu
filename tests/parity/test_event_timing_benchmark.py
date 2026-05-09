import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "event_timing_benchmark"


def test_event_timing_benchmark_correctness():
    """Use Event.elapsed_time(ev_start, ev_end) to time a launch."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    cfg.n_sm = 8

    ptx = (_DIR / "kernel.ptx").read_text()
    s = Stream()
    ev_start = Event()
    ev_end = Event()

    s.record(ev_start)
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
             params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add", config=cfg)
    s.record(ev_end)

    multi_res = gpusim.synchronize(streams=[s], config=cfg)

    np.testing.assert_array_equal(C, A + B)
    elapsed = Event.elapsed_time(ev_start, ev_end)
    assert isinstance(elapsed, int)
    assert elapsed >= 0
