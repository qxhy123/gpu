import numpy as np
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    buf = np.arange(32, dtype=np.float32)
    print("Pipeline forward pass:")
    for rank in range(3):
        comm = Comm(rank=rank, world_size=4, system=sys)
        cycles = comm.send(buf, dst_rank=rank + 1)
        print(f"  Rank {rank} → Rank {rank+1}: {cycles} cycles")
        comm_recv = Comm(rank=rank + 1, world_size=4, system=sys)
        comm_recv.recv(buf, src_rank=rank)


if __name__ == "__main__":
    main()
