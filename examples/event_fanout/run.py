import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    SHARED = np.zeros(n, dtype=np.uint32)
    outs = [np.zeros(n, dtype=np.uint32) for _ in range(3)]
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s_a = Stream()
    streams = [Stream() for _ in range(3)]
    ev = Event()
    s_a.launch(ptx_src=(here / "kernel_write.ptx").read_text(),
               grid=(1, 1, 1), block=(32, 1, 1),
               params={"OUT": SHARED}, kernel_name="write", config=cfg)
    s_a.record(ev)
    for s, out in zip(streams, outs):
        s.wait(ev)
        s.launch(ptx_src=(here / "kernel_read.ptx").read_text(),
                 grid=(1, 1, 1), block=(32, 1, 1),
                 params={"IN": SHARED, "OUT": out},
                 kernel_name=f"read_{s.stream_id}", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_a] + streams, config=cfg)
    print(multi_res.stream_summary())


if __name__ == "__main__":
    main()
