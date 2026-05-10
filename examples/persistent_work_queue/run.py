import numpy as np
import pathlib
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    queue = WorkQueue()
    out_bufs = []
    for _ in range(4):
        ob = np.zeros(32, dtype=np.uint32)
        out_bufs.append(ob)
        queue.push({"OUT": ob})
    queue.stop()
    pk = PersistentKernel(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                          params_template={}, work_queue=queue,
                          kernel_name="server")
    results = pk.start(cfg)
    print(f"Persistent kernel processed {len(results)} items")
    print(f"First buffer sum: {out_bufs[0].sum()} (expected 32)")


if __name__ == "__main__":
    main()
