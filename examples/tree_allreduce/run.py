import numpy as np
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default
from gpusim.trace.recorder import Recorder


def main():
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    rec = Recorder()
    comm = Comm(rank=0, world_size=4, system=sys)
    comm._recorder = rec
    send = np.full(16, 1.0, dtype=np.float32)
    recv = np.zeros(16, dtype=np.float32)
    cycles = comm.allreduce(send, recv, op="sum")
    print(f"Tree allreduce: {cycles} cycles")
    print(f"Algorithm chosen: {rec.collective_events[-1].algorithm}")
    print(f"recv[0:4] = {list(recv[0:4])}")


if __name__ == "__main__":
    main()
