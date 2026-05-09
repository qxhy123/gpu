import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "multi_gpu_setup"


def test_multi_gpu_setup_correctness():
    """2-GPU minimal demo: each GPU runs vec_add independently."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    OUT_0 = np.zeros(n, dtype=np.float32)
    OUT_1 = np.zeros(n, dtype=np.float32)

    cfg = load_default()
    cfg.n_gpus = 2

    sys = MultiGpuSystem.from_config(cfg)
    assert len(sys.gpus) == 2
    assert sys.nvlink_fabric is not None
    assert len(sys.nvlink_fabric.links) == 2

    ptx = (_DIR / "kernel.ptx").read_text()
    for gpu_idx, out in [(0, OUT_0), (1, OUT_1)]:
        gpu = sys.gpus[gpu_idx]
        s = Stream()
        s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                 params={"A": A, "B": B, "OUT": out},
                 kernel_name=f"vec_add_gpu{gpu_idx}", config=cfg)
        gpu.run_streams([s])

    np.testing.assert_array_equal(OUT_0, A + B)
    np.testing.assert_array_equal(OUT_1, A + B)
