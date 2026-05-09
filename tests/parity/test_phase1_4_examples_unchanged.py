"""Smoke-test: each Phase 1-4 example runs without crashing on Phase 5 Device path."""
import pytest
import pathlib, subprocess, sys


PHASE_1_4_EXAMPLES = [
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
]

# Examples that take > 2 min on the simulator; skipped in the fast suite.
SLOW_EXAMPLES = {"l1_thrash_demo"}


@pytest.mark.parametrize("ex", [e for e in PHASE_1_4_EXAMPLES if e not in SLOW_EXAMPLES])
def test_phase_1_4_example_smoke(ex):
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
def test_phase_1_4_example_smoke_slow(ex):
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
