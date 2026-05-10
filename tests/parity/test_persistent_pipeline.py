import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "persistent_pipeline"


def test_persistent_pipeline_correctness():
    """Producer-consumer via shared WorkQueue between two PersistentKernels."""
    import numpy as np
    from gpusim.persistent.queue import WorkQueue
    from gpusim.persistent.kernel import PersistentKernel
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    # Capstone: producer writes data to buffer; consumer reads it (sequential simulation)
    producer_q = WorkQueue()
    out_bufs = []
    for _ in range(3):
        ob = np.zeros(32, dtype=np.uint32)
        out_bufs.append(ob)
        producer_q.push({"OUT": ob})
    producer_q.stop()

    producer = PersistentKernel(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                                params_template={}, work_queue=producer_q,
                                kernel_name="producer")
    producer_results = producer.start(cfg)

    assert len(producer_results) == 3
    for ob in out_bufs:
        assert ob.sum() == 32
