import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "persistent_kernel_server"


def test_persistent_kernel_server_correctness():
    """Persistent kernel processes 5 work items."""
    import gpusim
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    queue = WorkQueue()
    out_bufs = []
    for _ in range(5):
        ob = np.zeros(32, dtype=np.uint32)
        out_bufs.append(ob)
        queue.push({"OUT": ob})
    queue.stop()

    pk = PersistentKernel(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                          params_template={}, work_queue=queue,
                          kernel_name="server")
    results = pk.start(cfg)

    assert len(results) == 5
    for ob in out_bufs:
        assert ob.sum() == 32   # each thread writes 1
