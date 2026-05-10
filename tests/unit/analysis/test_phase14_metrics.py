"""Unit tests for Phase 14 metrics: persistent_kernel_throughput,
dynamic_parallelism_depth, dynamic_parallelism_fanout."""
import pandas as pd


def test_persistent_kernel_throughput():
    from gpusim.analysis.metrics import persistent_kernel_throughput
    df = pd.DataFrame([
        {"is_persistent": True, "stream_id": 0, "parent_kernel_id": -1},
        {"is_persistent": True, "stream_id": 1, "parent_kernel_id": -1},
        {"is_persistent": False, "stream_id": 2, "parent_kernel_id": -1},
    ])
    rate = persistent_kernel_throughput(df, total_cycles=1000)
    # 2 persistent items / 1000 cycles * 1000 = 2.0
    assert abs(rate - 2.0) < 0.01


def test_dynamic_parallelism_fanout():
    from gpusim.analysis.metrics import dynamic_parallelism_fanout
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
        {"stream_id": 1, "parent_kernel_id": 0},   # child of 0
        {"stream_id": 2, "parent_kernel_id": 0},   # child of 0
        {"stream_id": 3, "parent_kernel_id": 1},   # child of 1
    ])
    out = dynamic_parallelism_fanout(df)
    assert out[0] == 2
    assert out[1] == 1


def test_dynamic_parallelism_depth():
    from gpusim.analysis.metrics import dynamic_parallelism_depth
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
        {"stream_id": 1, "parent_kernel_id": 0},
        {"stream_id": 2, "parent_kernel_id": 1},
        {"stream_id": 3, "parent_kernel_id": 2},
    ])
    depth = dynamic_parallelism_depth(df)
    # Chain: 0 -> 1 -> 2 -> 3 = depth 4 (or 3 children deep)
    assert depth >= 3


def test_persistent_kernel_throughput_empty():
    from gpusim.analysis.metrics import persistent_kernel_throughput
    df = pd.DataFrame(columns=["is_persistent", "stream_id", "parent_kernel_id"])
    assert persistent_kernel_throughput(df, total_cycles=1000) == 0.0


def test_persistent_kernel_throughput_zero_cycles():
    from gpusim.analysis.metrics import persistent_kernel_throughput
    df = pd.DataFrame([{"is_persistent": True, "stream_id": 0, "parent_kernel_id": -1}])
    assert persistent_kernel_throughput(df, total_cycles=0) == 0.0


def test_dynamic_parallelism_depth_empty():
    from gpusim.analysis.metrics import dynamic_parallelism_depth
    df = pd.DataFrame(columns=["stream_id", "parent_kernel_id"])
    assert dynamic_parallelism_depth(df) == 0


def test_dynamic_parallelism_depth_single_level():
    """Single root kernel (no parents) has depth == 1."""
    from gpusim.analysis.metrics import dynamic_parallelism_depth
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
    ])
    depth = dynamic_parallelism_depth(df)
    assert depth == 1


def test_dynamic_parallelism_fanout_empty():
    from gpusim.analysis.metrics import dynamic_parallelism_fanout
    df = pd.DataFrame(columns=["stream_id", "parent_kernel_id"])
    assert dynamic_parallelism_fanout(df) == {}


def test_dynamic_parallelism_fanout_no_children():
    from gpusim.analysis.metrics import dynamic_parallelism_fanout
    df = pd.DataFrame([
        {"stream_id": 0, "parent_kernel_id": -1},
        {"stream_id": 1, "parent_kernel_id": -1},
    ])
    out = dynamic_parallelism_fanout(df)
    assert out == {}
