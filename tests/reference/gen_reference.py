"""Run on a real NVIDIA GPU to produce *.ref.json fixtures for the simulator
to compare against.

Usage (on a CUDA-capable host):
    python tests/reference/gen_reference.py vector_add reduction_smem ...

Generates tests/reference/data/<kernel>.ref.json for each requested kernel.
The script depends on `nvcc` for compilation and (optionally) `ncu --csv`
for metrics. If `ncu` is unavailable, metrics are populated as best-effort
from CUPTI Python (cuda-python pkg) or skipped.
"""
from __future__ import annotations
import base64, json, sys, subprocess, pathlib, io
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / "data"


def encode_npy(arr: np.ndarray) -> str:
    bio = io.BytesIO()
    np.save(bio, arr, allow_pickle=False)
    return base64.b64encode(bio.getvalue()).decode("ascii")


def _run_nvcc_and_capture_outputs(kernel: str) -> dict:
    """Compile examples/<kernel>/kernel.cu and run with the same inputs as run.py.
    Returns dict with outputs (numpy arrays) and inputs (seed/shape).
    """
    raise NotImplementedError(
        "Implement on the GPU host: compile kernel.cu with nvcc, link a tiny "
        "host driver that mirrors examples/<kernel>/run.py inputs, and dump "
        "outputs into a numpy file."
    )


def _capture_metrics_via_ncu(kernel: str, exe: Path) -> dict:
    """Optionally call `ncu --csv --metrics ...` to collect achieved_occupancy,
    smsp__warps_active, etc. Returns dict subset matching the schema.
    """
    return {}


def gen(kernel: str) -> None:
    rec = {
        "kernel": kernel,
        "ptx_path": f"examples/{kernel}/kernel.ptx",
        "launch": {"grid": [1], "block": [32]},  # to be overridden by per-kernel logic
        "device": {"name": "H100 SXM5", "sm_count": 132},
        "inputs_shape": {},
        "inputs_seed": 42,
        "outputs": {},
        "metrics": {},
    }
    # subclass / monkeypatch this function per kernel; minimal stub here:
    DATA.mkdir(exist_ok=True)
    out = DATA / f"{kernel}.ref.json"
    out.write_text(json.dumps(rec, indent=2))
    print(f"wrote stub {out}")


SUPPORTED_KERNELS = [
    # Phase 1 kernels
    "vector_add",
    "reduction_smem",
    "tiled_matmul",
    "divergence_demo",
    "bank_conflict_demo",
    "coalescing_demo",
    # Phase 2 additions
    "l1_thrash_demo",
    "smem_vs_l1_demo",       # both variants share the same schema
    "bw_saturation_demo",
    "row_buffer_demo",
    # Phase 3 additions
    "tc_matmul_precisions",
    "mixed_accum",
    "wgmma_basic",
    "wgmma_async_pipeline",
    # Phase 4 additions
    "multi_sm_scheduler",
    "l2_sharing_demo",
    "tma_store_matmul",
    # Phase 5 additions
    "cluster_basic",
    "cluster_matmul_dsmem",
    "cluster_tma_pipeline",
    # Phase 6 additions
    "atom_histogram",
    "atom_reduction_smem",
    "cluster_cooperative_epilogue",
    "atom_cas_spinlock",
    "red_min_max",
    # Phase 7 additions
    "concurrent_vector_add_2stream",
    "compute_vs_memory_overlap",
    "l2_contention_2stream",
    "stream_priority_serial_vs_concurrent",
    # Phase 8 additions
    "true_concurrent_overlap",
    "priority_demo",
    "event_producer_consumer",
    "event_fanout",
    "l2_window_demo",
    "multi_stream_pipeline_full",
    # Phase 9 additions
    "phase8_overlap_real",
    "multi_event_fan_in",
    "event_timing_benchmark",
    # Phase 10 additions
    "multi_gpu_setup",
    "ring_allreduce",
    "tree_allreduce",
    "ddp_training_step",
]


def main(argv):
    if not argv:
        print("usage: gen_reference.py <kernel> [<kernel>...]")
        print("supported kernels:", ", ".join(SUPPORTED_KERNELS))
        return 2
    for k in argv:
        if k not in SUPPORTED_KERNELS:
            print(f"warning: {k!r} not in SUPPORTED_KERNELS; generating stub anyway")
        gen(k)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
