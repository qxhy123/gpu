import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "priority_demo"


def test_priority_demo_correctness():
    """3 streams (high/normal/low) each launch vec_add; all outputs correct."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2

    cfg = load_default()
    cfg.n_sm = 8

    ptx = (_DIR / "kernel.ptx").read_text()
    s_high = Stream(priority="high")
    s_normal = Stream(priority="normal")
    s_low = Stream(priority="low")

    out_h = np.zeros(n, dtype=np.float32)
    out_n = np.zeros(n, dtype=np.float32)
    out_l = np.zeros(n, dtype=np.float32)

    s_high.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                  params={"A": A, "B": B, "OUT": out_h}, kernel_name="kh", config=cfg)
    s_normal.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                    params={"A": A, "B": B, "OUT": out_n}, kernel_name="kn", config=cfg)
    s_low.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                 params={"A": A, "B": B, "OUT": out_l}, kernel_name="kl", config=cfg)

    multi_res = gpusim.synchronize(streams=[s_high, s_normal, s_low], config=cfg)

    np.testing.assert_array_equal(out_h, A + B)
    np.testing.assert_array_equal(out_n, A + B)
    np.testing.assert_array_equal(out_l, A + B)
    assert len(multi_res.streams) == 3
