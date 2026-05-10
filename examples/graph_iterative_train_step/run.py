import numpy as np
import pathlib
from gpusim.api import Stream
from gpusim.config.loader import load_default


def main():
    n = 32
    weights = np.zeros(n, dtype=np.float32)
    grads = np.ones(n, dtype=np.float32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    s = Stream()
    s.begin_capture()
    s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
              params={"WEIGHTS": weights, "GRADS": grads},
              kernel_name="sgd_update", config=cfg)
    g = s.end_capture()
    exec = g.instantiate(cfg)
    for epoch in range(3):
        exec.launch()
    print(f"After 3 epochs, weights[0:4]: {list(weights[0:4])}")


if __name__ == "__main__":
    main()
