import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    C = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    ev_start = Event(); ev_end = Event()
    s.record(ev_start)
    s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
             params={"A": A, "B": B, "OUT": C}, kernel_name="vec_add", config=cfg)
    s.record(ev_end)
    gpusim.synchronize(streams=[s], config=cfg)
    print(f"Kernel took {Event.elapsed_time(ev_start, ev_end)} cycles")


if __name__ == "__main__":
    main()
