import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "stream_capture_multi_stream"


def test_stream_capture_multi_stream_correctness():
    """sA: k1 → record(ev) → k3.  sB: wait(ev) → k2.  Captured into one graph."""
    from gpusim.graph.capture_session import CaptureSession
    from gpusim.api import Stream, Event
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    OUT = np.zeros(32, dtype=np.uint32)

    sess = CaptureSession()
    sA = Stream()
    sB = Stream()
    sess.attach(sA)
    sess.attach(sB)
    ev = Event()

    sA.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k1", config=cfg)
    sA.record(ev)
    sB.wait(ev)
    sB.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k2", config=cfg)
    sA.launch(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
               params={"OUT": OUT}, kernel_name="k3", config=cfg)

    g = sess.end()
    # 5 nodes: k1, record, wait, k2, k3
    assert len(g.nodes) == 5
    # Edges: k1→record, record→wait (cross-stream), wait→k2, record→k3 (intra-stream A chain)
    assert len(g.edges) >= 3   # at minimum: k1→record, record→wait (cross-stream), wait→k2

    exec = g.instantiate(cfg)
    exec.launch()
    # 3 kernels * 32 threads * 1 increment = 96
    assert OUT.sum() == 96
