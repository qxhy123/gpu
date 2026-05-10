import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_iterative_train_step"


def test_graph_iterative_train_step_correctness():
    """Capture training step graph; replay 3 times to simulate 3 epochs."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    weights = np.zeros(n, dtype=np.float32)
    grads = np.ones(n, dtype=np.float32)

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
              params={"WEIGHTS": weights, "GRADS": grads},
              kernel_name="sgd_update", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)

    for epoch in range(3):
        exec.launch()

    np.testing.assert_array_equal(weights, np.full(n, -3.0, dtype=np.float32))
