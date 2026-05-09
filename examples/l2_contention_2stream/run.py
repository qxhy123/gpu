import numpy as np, pathlib, gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    SHARED = np.zeros(64, dtype=np.uint32)
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s0 = Stream()
    s1 = Stream()
    s0.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 0}, kernel_name="writer_low", config=cfg)
    s1.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
              params={"OUT": SHARED, "OFFSET": 32}, kernel_name="writer_high", config=cfg)
    multi_res = gpusim.synchronize(streams=[s0, s1], config=cfg)
    print(multi_res.stream_summary())


if __name__ == "__main__":
    main()
