"""Smoke-test: each Phase 1-7 example runs without crashing on Phase 7 Device path."""
import pytest
import pathlib, subprocess, sys


PHASE_1_7_EXAMPLES = [
    # Phase 1
    "vector_add",
    "reduction_smem",
    "tiled_matmul",
    "divergence_demo",
    "bank_conflict_demo",
    "coalescing_demo",
    # Phase 2
    "l1_thrash_demo",
    "smem_vs_l1_demo",
    "bw_saturation_demo",
    "row_buffer_demo",
    # Phase 3
    "tc_matmul_precisions",
    "mixed_accum",
    "wgmma_basic",
    "wgmma_async_pipeline",
    # Phase 4
    "multi_sm_scheduler",
    "l2_sharing_demo",
    "tma_store_matmul",
    # Phase 5
    "cluster_basic",
    "cluster_matmul_dsmem",
    "cluster_tma_pipeline",
    # Phase 6
    "atom_histogram",
    "atom_reduction_smem",
    "atom_cas_spinlock",
    "red_min_max",
    "cluster_cooperative_epilogue",
    # Phase 7
    "concurrent_vector_add_2stream",
    "compute_vs_memory_overlap",
    "l2_contention_2stream",
    "stream_priority_serial_vs_concurrent",
]

# Examples that take > 2 min on the simulator; skipped in the fast suite.
SLOW_EXAMPLES = {"l1_thrash_demo"}


@pytest.mark.parametrize("ex", [e for e in PHASE_1_7_EXAMPLES if e not in SLOW_EXAMPLES])
def test_phase_1_7_example_smoke(ex):
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / ex
    if not (base / "run.py").exists():
        pytest.skip(f"no run.py for {ex}")
    res = subprocess.run(
        [sys.executable, str(base / "run.py")],
        capture_output=True, timeout=120,
    )
    assert res.returncode == 0, (
        f"{ex}/run.py failed (rc={res.returncode}): "
        f"stderr={res.stderr.decode()[-500:]}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("ex", sorted(SLOW_EXAMPLES))
def test_phase_1_7_example_smoke_slow(ex):
    """Same smoke test for examples that exceed the 120 s fast-suite timeout."""
    base = pathlib.Path(__file__).resolve().parents[2] / "examples" / ex
    if not (base / "run.py").exists():
        pytest.skip(f"no run.py for {ex}")
    res = subprocess.run(
        [sys.executable, str(base / "run.py")],
        capture_output=True, timeout=600,
    )
    assert res.returncode == 0, (
        f"{ex}/run.py failed (rc={res.returncode}): "
        f"stderr={res.stderr.decode()[-500:]}"
    )
