import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s_a = Stream(); s_b = Stream()
    ev = Event()
    s_a.launch(ptx_src=(here / "kernel_write.ptx").read_text(),
               grid=(1, 1, 1), block=(32, 1, 1),
               params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    s_b.wait(ev)
    s_b.launch(ptx_src=(here / "kernel_read.ptx").read_text(),
               grid=(1, 1, 1), block=(32, 1, 1),
               params={"IN": SHARED, "OUT": OUT}, kernel_name="read", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_a, s_b], config=cfg)
    print(multi_res.stream_summary())
    print(f"Event wait cycles: {multi_res.event_wait_cycles_per_stream()}")


if __name__ == "__main__":
    main()
