"""Phase 10 microbench — multi-GPU + NCCL facts."""
import numpy as np


def test_ring_allreduce_step_count():
    """Ring allreduce: 2*(N-1) NVLink transfers."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    # Hook recorder into fabric for transfer events
    original_transfer = sys.nvlink_fabric.transfer
    def traced_transfer(*args, **kwargs):
        kwargs["recorder"] = rec
        return original_transfer(*args, **kwargs)
    sys.nvlink_fabric.transfer = traced_transfer

    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(1024, 1.0, dtype=np.float32)   # 4096 bytes — ring path
    recv = np.zeros(1024, dtype=np.float32)
    comm.allreduce(send, recv, op="sum")

    # Should have 2*(4-1) = 6 transfers
    assert len(rec.nvlink_transfer_events) == 6


def test_tree_allreduce_step_count():
    """Tree allreduce: 2*log2(N) NVLink transfers."""
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
    send = np.full(8, 1.0, dtype=np.float32)   # 32 bytes — tree path
    recv = np.zeros(8, dtype=np.float32)
    comm.allreduce(send, recv, op="sum")

    # 2*log2(4) = 4 transfers
    assert len(rec.nvlink_transfer_events) == 4


def test_tree_picked_for_small_msg():
    """Tree algorithm chosen for messages < 4096 bytes."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(8, 1.0, dtype=np.float32)   # 32 bytes
    recv = np.zeros(8, dtype=np.float32)
    comm.allreduce(send, recv, op="sum")
    assert rec.collective_events[-1].algorithm == "tree"
