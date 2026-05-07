import numpy as np
import pathlib
import gpusim

PTX = (pathlib.Path(__file__).parents[3] / "examples/divergence_demo/kernel.ptx").read_text()


def test_divergent_path_records_divergence_serial_state():
    out = np.zeros(32, dtype=np.uint32)
    res = gpusim.run(ptx_src=PTX, grid=(1,1,1), block=(32,1,1),
                     params={"OUT": out}, mode="timing")
    states = res.events_df.groupby("state")["end"].count().to_dict()
    assert "DIVERGENCE_SERIAL" in states, \
        f"expected DIVERGENCE_SERIAL in trace states, got {list(states)}"
