import numpy as np
import pathlib
import gpusim
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32) * 2
    cfg = load_default()
    cfg.n_sm = 8
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s_high = Stream(priority="high")
    s_normal = Stream(priority="normal")
    s_low = Stream(priority="low")
    out_h = np.zeros(n, dtype=np.float32)
    out_n = np.zeros(n, dtype=np.float32)
    out_l = np.zeros(n, dtype=np.float32)
    for s, out, name in [(s_high, out_h, "kh"), (s_normal, out_n, "kn"), (s_low, out_l, "kl")]:
        s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
                 params={"A": A, "B": B, "OUT": out}, kernel_name=name, config=cfg)
    multi_res = gpusim.synchronize(streams=[s_high, s_normal, s_low], config=cfg)
    print(multi_res.stream_summary())
    print(f"Priority dispatch share: {multi_res.priority_dispatch_share()}")


if __name__ == "__main__":
    main()
