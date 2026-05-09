import pathlib, numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "ddp_training_step"


def test_ddp_training_step_correctness():
    """4-GPU DDP-style: vec_add → allreduce gradients → broadcast weights."""
    import gpusim
    from gpusim.api import Stream, _reset_stream_id_counter
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    _reset_stream_id_counter()

    n = 32
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)

    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    grads = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)

    ptx = (_DIR / "kernel.ptx").read_text()
    for rank in range(4):
        gpu = sys.gpus[rank]
        s = Stream()
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": grads},
                  kernel_name=f"compute_grad_rank{rank}", config=cfg)
        gpu.run_streams([s])

    comm0 = Comm(rank=0, world_size=4, system=sys)
    grads_reduced = np.zeros(n, dtype=np.float32)
    cycles_ar = comm0.allreduce(grads, grads_reduced, op="sum")
    cycles_bc = comm0.broadcast(weights, root=0)

    assert cycles_ar > 0
    assert cycles_bc > 0
    np.testing.assert_array_equal(grads_reduced, grads * 4)
