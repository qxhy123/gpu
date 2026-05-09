import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_stream_pipeline_full"


def test_multi_stream_pipeline_full_correctness():
    """3 streams (load → compute → store) with priority + events + L2 window."""
    import gpusim
    from gpusim.api import Stream, Event, _reset_stream_id_counter, _reset_event_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter(); _reset_event_id_counter()

    n = 32
    A = np.arange(n, dtype=np.float32)
    INTER = np.zeros(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    cfg.n_sm = 8

    s_load = Stream(priority="normal")
    s_compute = Stream(priority="high")
    s_compute.set_l2_window(start_set=0, n_sets=16)
    s_store = Stream(priority="normal")

    ev_load_done = Event()
    ev_compute_done = Event()

    here = _DIR
    s_load.launch(ptx_src=(here / "kernel_load.ptx").read_text(),
                   grid=(1,1,1), block=(32,1,1),
                   params={"IN": A, "OUT": INTER}, kernel_name="load", config=cfg)
    s_load.record(ev_load_done)

    s_compute.wait(ev_load_done)
    s_compute.launch(ptx_src=(here / "kernel_compute.ptx").read_text(),
                       grid=(1,1,1), block=(32,1,1),
                       params={"IN": INTER, "OUT": INTER}, kernel_name="compute", config=cfg)
    s_compute.record(ev_compute_done)

    s_store.wait(ev_compute_done)
    s_store.launch(ptx_src=(here / "kernel_store.ptx").read_text(),
                    grid=(1,1,1), block=(32,1,1),
                    params={"IN": INTER, "OUT": OUT}, kernel_name="store", config=cfg)

    multi_res = gpusim.synchronize(streams=[s_load, s_compute, s_store], config=cfg)

    expected = A * 2.0
    np.testing.assert_array_almost_equal(OUT, expected)
    assert len(multi_res.streams) == 3
