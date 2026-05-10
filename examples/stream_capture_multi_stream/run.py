import numpy as np, pathlib
from gpusim.graph.capture_session import CaptureSession
from gpusim.api import Stream, Event
from gpusim.config.loader import load_default


def main():
    n = 32
    OUT = np.zeros(n, dtype=np.uint32)
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()

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
    print(f"Captured: {len(g.nodes)} nodes, {len(g.edges)} edges")
    print(f"Edges: {g.edges}")

    exec = g.instantiate(cfg)
    exec.launch()
    print(f"OUT.sum() = {OUT.sum()} (expected 96)")


if __name__ == "__main__":
    main()
