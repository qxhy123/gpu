import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    INTER = np.zeros(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s_load = Stream(priority="normal")
    s_compute = Stream(priority="high")
    s_compute.set_l2_window(start_set=0, n_sets=16)
    s_store = Stream(priority="normal")
    ev1 = Event(); ev2 = Event()
    s_load.launch(ptx_src=(here / "kernel_load.ptx").read_text(),
                   grid=(1,1,1), block=(32,1,1),
                   params={"IN": A, "OUT": INTER}, kernel_name="load", config=cfg)
    s_load.record(ev1)
    s_compute.wait(ev1)
    s_compute.launch(ptx_src=(here / "kernel_compute.ptx").read_text(),
                       grid=(1,1,1), block=(32,1,1),
                       params={"IN": INTER, "OUT": INTER}, kernel_name="compute", config=cfg)
    s_compute.record(ev2)
    s_store.wait(ev2)
    s_store.launch(ptx_src=(here / "kernel_store.ptx").read_text(),
                    grid=(1,1,1), block=(32,1,1),
                    params={"IN": INTER, "OUT": OUT}, kernel_name="store", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_load, s_compute, s_store], config=cfg)
    print(multi_res.stream_summary())
    print(f"OUT[0:4] = {list(OUT[0:4])}")
    print(f"event_chain_critical_path: {multi_res.event_chain_critical_path()}")


if __name__ == "__main__":
    main()
