import numpy as np
import pathlib
from gpusim.persistent.dynamic import device_launch, drain_pending_child_launches
from gpusim.api import Stream, synchronize
from gpusim.config.loader import load_default


def main():
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    out_a = np.zeros(32, dtype=np.uint32)
    out_b = np.zeros(32, dtype=np.uint32)
    out_c = np.zeros(32, dtype=np.uint32)

    s = Stream()
    s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
             params={"OUT": out_a}, kernel_name="parent", config=cfg)
    synchronize(streams=[s], config=cfg)
    print(f"Parent ran: out_a sum = {out_a.sum()}")

    # Parent triggers child
    device_launch(parent_kernel_id=s.stream_id, ptx_src=ptx,
                  grid=(1, 1, 1), block=(32, 1, 1),
                  params={"OUT": out_b}, kernel_name="child")
    drain_pending_child_launches(cfg)
    print(f"Child ran: out_b sum = {out_b.sum()}")

    # Child triggers grandchild
    device_launch(parent_kernel_id=s.stream_id + 1, ptx_src=ptx,
                  grid=(1, 1, 1), block=(32, 1, 1),
                  params={"OUT": out_c}, kernel_name="grandchild")
    drain_pending_child_launches(cfg)
    print(f"Grandchild ran: out_c sum = {out_c.sum()}")


if __name__ == "__main__":
    main()
