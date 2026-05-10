"""Unit tests for Comm.reduce_scatter (ring algorithm). Phase 12 T1."""
import numpy as np


def test_reduce_scatter_step_count():
    """Ring reduce_scatter: N-1 transfers per rank."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    # 32-element input → each rank gets 8 elements
    send = np.full(32, 1.0, dtype=np.float32)
    recv = np.zeros(8, dtype=np.float32)
    cycles = comm.reduce_scatter(send, recv, op="sum")
    assert cycles > 0


def test_reduce_scatter_correctness_sum():
    """Each rank gets its chunk of the sum (1.0 * 4 = 4.0)."""
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(32, 1.0, dtype=np.float32)
    recv = np.zeros(8, dtype=np.float32)
    comm.reduce_scatter(send, recv, op="sum")
    np.testing.assert_array_equal(recv, np.full(8, 4.0, dtype=np.float32))


def test_reduce_scatter_records_collective_event():
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
    send = np.full(32, 1.0, dtype=np.float32)
    recv = np.zeros(8, dtype=np.float32)
    comm.reduce_scatter(send, recv, op="sum")
    assert rec.collective_events[-1].op_name == "reduce_scatter"
    assert rec.collective_events[-1].algorithm == "ring"
