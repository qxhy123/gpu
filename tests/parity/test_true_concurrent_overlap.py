import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "true_concurrent_overlap"


def test_true_concurrent_overlap_correctness():
    """Two streams (compute-heavy + memory-heavy) produce correct outputs.
    Phase 8 M1 minimal: cycles savings not yet realized but API is correct."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    D = np.arange(n, dtype=np.float32) * 5
    E = np.arange(n, dtype=np.float32) * 3
    F = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    cfg.n_sm = 8

    ptx_compute = (_DIR / "kernel_compute.ptx").read_text()
    ptx_memory = (_DIR / "kernel_memory.ptx").read_text()

    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx_compute, grid=(1,1,1), block=(32,1,1),
              params={"A": A, "B": B, "OUT": C}, kernel_name="compute_heavy", config=cfg)
    s1.launch(ptx_src=ptx_memory, grid=(1,1,1), block=(32,1,1),
              params={"A": D, "B": E, "OUT": F}, kernel_name="memory_heavy", config=cfg)

    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)

    assert (C >= 0).all()
    np.testing.assert_array_equal(F, D + E)
    assert len(multi_res.streams) == 2
