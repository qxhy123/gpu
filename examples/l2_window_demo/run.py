import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    OUT_HIGH = np.zeros(n, dtype=np.uint32)
    OUT_LOW = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s_high = Stream(priority="high")
    s_high.set_l2_window(start_set=0, n_sets=32)
    s_low = Stream(priority="low")
    s_high.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                    params={"OUT": OUT_HIGH}, kernel_name="critical", config=cfg)
    s_low.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                   params={"OUT": OUT_LOW}, kernel_name="background", config=cfg)
    multi_res = gpusim.synchronize(streams=[s_high, s_low], config=cfg)
    print(multi_res.stream_summary())


if __name__ == "__main__":
    main()
