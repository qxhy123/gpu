import pathlib
import numpy as np


_DIR = pathlib.Path(__file__).resolve().parents[2] / "examples" / "graph_while_loop"


def test_graph_while_loop_correctness():
    """Loop until counter reaches 0; body runs N iterations."""
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default

    cfg = load_default()
    ptx = (_DIR / "kernel.ptx").read_text()
    OUT = np.zeros(32, dtype=np.uint32)

    counter = [4]
    body = Graph()
    body.add_kernel_node(ptx_src=ptx, grid=(1,1,1), block=(32,1,1),
                            params={"OUT": OUT}, kernel_name="inc")

    def cond():
        if counter[0] > 0:
            counter[0] -= 1
            return True
        return False

    g = Graph()
    g.add_while_node(cond_fn=cond, body_graph=body, max_iterations=10)
    g.instantiate(cfg).launch()
    # 4 iterations × 32 threads × 1 increment = 128
    assert OUT.sum() == 128
