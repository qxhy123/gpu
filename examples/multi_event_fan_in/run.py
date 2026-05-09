import numpy as np, pathlib, gpusim
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.zeros(n, dtype=np.uint32)
    B = np.zeros(n, dtype=np.uint32)
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    here = pathlib.Path(__file__).parent
    s_a = Stream(); s_b = Stream(); s_c = Stream()
    ev_a = Event(); ev_b = Event()
    s_a.launch(ptx_src=(here / "kernel_write.ptx").read_text(),
               grid=(1,1,1), block=(32,1,1),
               params={"OUT": A}, kernel_name="write_a", config=cfg)
    s_a.record(ev_a)
    s_b.launch(ptx_src=(here / "kernel_write.ptx").read_text(),
               grid=(1,1,1), block=(32,1,1),
               params={"OUT": B}, kernel_name="write_b", config=cfg)
    s_b.record(ev_b)
    s_c.wait_all([ev_a, ev_b])
    s_c.launch(ptx_src=(here / "kernel_combine.ptx").read_text(),
               grid=(1,1,1), block=(32,1,1),
               params={"A": A, "B": B, "OUT": OUT}, kernel_name="combine", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_a, s_b, s_c], config=cfg)
    print(multi_res.stream_summary())
    print(f"OUT[0:4] = {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
