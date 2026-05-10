import numpy as np


def test_update_kernel_node_params():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k0")
    exec = g.instantiate(cfg)
    exec.update_kernel_node_params(0, kernel_name="k0_renamed")
    assert g.nodes[0].kernel_args.kernel_name == "k0_renamed"
    assert exec._update_count == 1


def test_update_invalid_node_raises():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    with pytest.raises(ValueError, match="not found"):
        exec.update_kernel_node_params(99, kernel_name="x")


def test_update_non_kernel_raises():
    import numpy as np
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    g = Graph()
    g.add_memset_node(buf=np.zeros(8, dtype=np.uint8), value=0, n_bytes=8)
    exec = g.instantiate(cfg)
    with pytest.raises(ValueError, match="not kernel"):
        exec.update_kernel_node_params(0, kernel_name="x")


def test_update_unknown_field_raises():
    from gpusim.graph.graph import Graph
    from gpusim.config.loader import load_default
    import pytest
    cfg = load_default()
    g = Graph()
    g.add_kernel_node(ptx_src="x", grid=(1,1,1), block=(32,1,1),
                        params={}, kernel_name="k")
    exec = g.instantiate(cfg)
    with pytest.raises(ValueError, match="unknown update field"):
        exec.update_kernel_node_params(0, bogus_field="x")
