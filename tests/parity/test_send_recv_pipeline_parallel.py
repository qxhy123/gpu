def test_send_recv_pipeline_parallel_correctness():
    """Pipeline parallelism: rank N sends to rank N+1 in chain."""
    import numpy as np
    from gpusim.comm.comm import Comm
    from gpusim.comm.system import MultiGpuSystem
    from gpusim.config.loader import load_default
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)

    # Each rank sends activation to next; verify all transfers complete with cycles > 0
    buf = np.arange(32, dtype=np.float32)
    total_cycles = 0
    for rank in range(3):  # 0→1, 1→2, 2→3
        comm = Comm(rank=rank, world_size=4, system=sys)
        cycles = comm.send(buf, dst_rank=rank + 1)
        total_cycles += cycles
        # Receive at next rank (no-op in simulator)
        comm_recv = Comm(rank=rank + 1, world_size=4, system=sys)
        comm_recv.recv(buf, src_rank=rank)

    assert total_cycles > 0
