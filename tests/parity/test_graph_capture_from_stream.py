import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_capture_from_stream"


def test_graph_capture_from_stream_correctness():
    """Capture 3-launch sequence into Graph; instantiate; launch."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    s = Stream()
    s.begin_capture()
    for i in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": OUT},
                  kernel_name=f"vec_add_{i}", config=cfg)
    g = s.end_capture()

    assert len(g.nodes) == 3
    assert len(g.edges) == 2

    exec = g.instantiate(cfg)
    cycles = exec.launch()

    np.testing.assert_array_equal(OUT, A + B)
    assert cycles > 0
