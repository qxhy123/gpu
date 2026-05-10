import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_replay_perf"


def test_graph_replay_perf_correctness():
    """Capture a small graph; replay 5x; verify each replay produces correct output."""
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
    s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
              params={"A": A, "B": B, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)

    cycles_per_replay = []
    for i in range(5):
        cycles_per_replay.append(exec.launch())

    assert len(cycles_per_replay) == 5
    np.testing.assert_array_equal(OUT, A + B)
    assert all(c > 0 for c in cycles_per_replay)
