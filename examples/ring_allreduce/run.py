import numpy as np
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    comm = Comm(rank=0, world_size=4, system=sys)
    send = np.full(256, 1.0, dtype=np.float32)
    recv = np.zeros(256, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")
    print(f"Ring allreduce: {cycles} cycles")
    print(f"recv[0:4] = {list(recv[0:4])}  (expected: [4.0, 4.0, 4.0, 4.0])")


if __name__ == "__main__":
    main()
