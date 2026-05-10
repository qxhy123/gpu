import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_update_replay"


def test_graph_update_replay_correctness():
    """Capture single-kernel graph; replay 3 times. Between replays, swap input buffers."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    A1 = np.full(n, 1.0, dtype=np.float32)
    B1 = np.full(n, 1.0, dtype=np.float32)
    A2 = np.full(n, 5.0, dtype=np.float32)
    B2 = np.full(n, 3.0, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A1, "B": B1, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)

    # Launch 1 with A1+B1
    exec.launch()
    np.testing.assert_array_equal(OUT, A1 + B1)

    # Update params to A2+B2
    exec.update_kernel_node_params(0, params={"A": A2, "B": B2, "OUT": OUT})
    exec.launch()
    np.testing.assert_array_equal(OUT, A2 + B2)

    assert exec._update_count == 1
