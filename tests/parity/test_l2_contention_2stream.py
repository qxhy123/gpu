import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "l2_contention_2stream"


def test_l2_contention_2stream_correctness():
    """Two streams write to overlapping gmem range; both regions written."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 64
    SHARED = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8

    ptx = (_DIR / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    # Both streams write to same buffer, different offsets within same L2 line range
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 0}, kernel_name="writer_low", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 32}, kernel_name="writer_high", config=cfg)

    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)

    # Both regions should have been written; expect non-zero
    assert SHARED[0:32].sum() > 0
    assert SHARED[32:64].sum() > 0
    assert len(multi_res.streams) == 2
