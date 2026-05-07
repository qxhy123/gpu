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


def main(argv):
    if not argv:
        print("usage: gen_reference.py <kernel> [<kernel>...]"); return 2
    for k in argv:
        gen(k)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
