import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "concurrent_vector_add_2stream"


def test_concurrent_vector_add_2stream_correctness():
    """Two streams each run vector_add on independent arrays; both outputs correct."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 3
    E = np.arange(n, dtype=np.float32) * 4
    F = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    cfg.n_sm = 8

    ptx = (_DIR / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add_a", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F}, kernel_name="vec_add_b", config=cfg)

    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)

    # Both outputs correct
    np.testing.assert_array_equal(C, A + B)
    np.testing.assert_array_equal(F, D + E)
    # Both streams produced one Result each
    assert len(multi_res.streams[0]) == 1
    assert len(multi_res.streams[1]) == 1
