import numpy as np, pathlib
from gpusim.api import Stream
from gpusim.comm.comm import Comm
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    grads = np.zeros(n, dtype=np.float32)
    weights = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_gpus = 4
    sys = MultiGpuSystem.from_config(cfg)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    for rank in range(4):
        s = Stream()
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"A": A, "B": B, "OUT": grads},
                  kernel_name=f"compute_rank{rank}", config=cfg)
        sys.gpus[rank].run_streams([s])
    comm0 = Comm(rank=0, world_size=4, system=sys)
    grads_reduced = np.zeros(n, dtype=np.float32)
    cycles_ar = comm0.allreduce(grads, grads_reduced, op="sum")
    cycles_bc = comm0.broadcast(weights, root=0)
    print(f"Allreduce cycles: {cycles_ar}")
    print(f"Broadcast cycles: {cycles_bc}")
    print(f"Reduced gradients[0:4]: {list(grads_reduced[0:4])}")


if __name__ == "__main__":
    main()
