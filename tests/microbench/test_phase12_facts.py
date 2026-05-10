"""Phase 12 microbench — NCCL completion facts."""
import numpy as np


def test_reduce_scatter_step_count_n_minus_1():
    """Ring reduce_scatter: N-1 NVLink transfers."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    original_transfer = sys.nvlink_fabric.transfer
    def traced_transfer(*args, **kwargs):
        kwargs["recorder"] = rec
        return original_transfer(*args, **kwargs)
    sys.nvlink_fabric.transfer = traced_transfer

    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(64, 1.0, dtype=np.float32)
    recv = np.zeros(16, dtype=np.float32)
    comm.reduce_scatter(send, recv, op="sum")

    assert len(rec.nvlink_transfer_events) == 3   # N-1 = 3 for N=4


def test_send_one_transfer():
    """Single send produces single NVLink transfer."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=2, system=sys)
    comm._recorder = rec
    buf = np.arange(64, dtype=np.float32)
    comm.send(buf, dst_rank=1)
    assert len(rec.nvlink_transfer_events) == 1
