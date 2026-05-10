import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_basic"


def test_stream_capture_basic_correctness():
    """Capture 3-kernel sequence + replay 5 times produces 5x output."""
    from gpusim.api import Stream
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    OUT = np.zeros(32, dtype=np.uint32)
    s = Stream()
    s.begin_capture()
    for _ in range(3):
        s.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                  params={"OUT": OUT}, kernel_name="inc", config=cfg)
    g = s.end_capture()

    assert g.is_captured is True
    assert len(g.nodes) == 3
    assert len(g.edges) == 2

    exec = g.instantiate(cfg)
    for _ in range(5):
        exec.launch()
    # 5 replays * 3 kernels per replay * 1 increment per thread per kernel = 15
    assert OUT.sum() == 32 * 15
