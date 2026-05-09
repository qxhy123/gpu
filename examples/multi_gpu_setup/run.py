import numpy as np
import pathlib
from gpusim.api import Stream
from gpusim.comm.system import MultiGpuSystem
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    OUT_0 = np.zeros(n, dtype=np.float32)
    OUT_1 = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_gpus = 2
    sys = MultiGpuSystem.from_config(cfg)
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    for i, out in enumerate([OUT_0, OUT_1]):
        s = Stream()
        s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                 params={"A": A, "B": B, "OUT": out},
                 kernel_name=f"vec_add_gpu{i}", config=cfg)
        sys.gpus[i].run_streams([s])
    print(f"GPU 0 output: {OUT_0[0:4]}")
    print(f"GPU 1 output: {OUT_1[0:4]}")
    print(f"NVLink topology: {len(sys.nvlink_fabric.links)} links across {sys.nvlink_fabric.n_gpus} GPUs")


if __name__ == "__main__":
    main()
