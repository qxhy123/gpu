import numpy as np
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    grads = np.full(64, 1.0, dtype=np.float32)
    my_chunk = np.zeros(16, dtype=np.float32)
    cycles = comm.reduce_scatter(grads, my_chunk, op="sum")
    print(f"Reduce_scatter: {cycles} cycles")
    print(f"Rank 0 chunk[0:4] = {list(my_chunk[0:4])} (expected [4.0, 4.0, 4.0, 4.0])")


if __name__ == "__main__":
    main()
