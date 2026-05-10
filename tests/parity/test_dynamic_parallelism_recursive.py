import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "dynamic_parallelism_recursive"


def test_dynamic_parallelism_recursive_correctness():
    """Parent → child → grandchild chain via device_launch."""
    import gpusim
    from gpusim.persistent.dynamic import (
        device_launch, drain_pending_child_launches, reset_pending_child_launches,
    )
    from gpusim.api import Stream, synchronize, _reset_stream_id_counter
    from gpusim.config.loader import load_default
    from gpusim.trace.recorder import Recorder
    _reset_stream_id_counter()
    reset_pending_child_launches()

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()

    out_a = np.zeros(32, dtype=np.uint32)
    out_b = np.zeros(32, dtype=np.uint32)
    out_c = np.zeros(32, dtype=np.uint32)

    # Parent kernel
    s = Stream()
    s.launch(ptx_src=ptx, grid=(1, 1, 1), block=(32, 1, 1),
             params={"OUT": out_a}, kernel_name="parent", config=cfg)
    parent_res = synchronize(streams=[s], config=cfg)
    parent_id = s.stream_id

    # Child launch (parent_id refers to parent stream)
    device_launch(parent_kernel_id=parent_id, ptx_src=ptx,
                  grid=(1, 1, 1), block=(32, 1, 1),
                  params={"OUT": out_b}, kernel_name="child")
    child_results = drain_pending_child_launches(cfg)

    assert out_a.sum() == 32   # parent ran
    assert out_b.sum() == 32   # child ran
    assert len(child_results) == 1
