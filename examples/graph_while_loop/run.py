import numpy as np, pathlib
from gpusim.graph.graph import Graph
from gpusim.config.loader import load_default


def main():
    n = 32
    cfg = load_default()
    ptx = (pathlib.Path(__file__).parent / "kernel.ptx").read_text()
    OUT = np.zeros(n, dtype=np.uint32)

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
    print(f"After while loop (4 iters): OUT.sum() = {OUT.sum()} (expected 128)")


if __name__ == "__main__":
    main()
