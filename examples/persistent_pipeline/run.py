import numpy as np
import pathlib
from gpusim.persistent.queue import WorkQueue
from gpusim.persistent.kernel import PersistentKernel
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

    # Stage 1: producer fills buffers
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
    print(f"Producer processed {len(producer_results)} items")

    # Stage 2: consumer verifies each buffer
    consumer_q = WorkQueue()
    for ob in out_bufs:
        consumer_q.push({"OUT": ob})
    consumer_q.stop()

    consumer = PersistentKernel(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                                params_template={}, work_queue=consumer_q,
                                kernel_name="consumer")
    consumer_results = consumer.start(cfg)
    print(f"Consumer processed {len(consumer_results)} items")
    print(f"Pipeline complete. First buffer sum: {out_bufs[0].sum()} (expected 32)")


if __name__ == "__main__":
    main()
