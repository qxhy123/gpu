import numpy as np
import pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    A = np.arange(n, dtype=np.float32)
    B = np.arange(n, dtype=np.float32)
    OUT = np.zeros(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
              params={"A": A, "B": B, "OUT": OUT},
              kernel_name="vec_add", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    cycles_per_replay = [exec.launch() for _ in range(5)]
    print(f"Replay cycles per launch: {cycles_per_replay}")
    print(f"Average cycles/replay: {sum(cycles_per_replay)/5:.1f}")
    print(f"Final OUT[0:4]: {list(OUT[0:4])}")


if __name__ == "__main__":
    main()
