"""Run every example in timing mode, emit HTML + Perfetto + summary stats.

Used to validate that Phase 1's teaching pipeline actually surfaces the
phenomena it's supposed to show. Outputs go to /tmp/gpusim-demo/.
"""
from __future__ import annotations
import pathlib, json, sys
import numpy as np
import gpusim

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = pathlib.Path("/tmp/gpusim-demo")
OUT.mkdir(exist_ok=True)


def _run(name, ptx_path, grid, block, params, regs_per_thread=16, smem_per_cta=0):
    ptx = ptx_path.read_text()
    res = gpusim.run(
        ptx_src=ptx, grid=grid, block=block,
        params=params, mode="timing",
    )
    # html / perfetto / parquet
    html = OUT / f"{name}.html"
    perf = OUT / f"{name}.json"
    res.html_report(html)
    res.perfetto(perf)
    return res, html, perf


def vector_add():
    n = 1024
    a = np.random.RandomState(0).randn(n).astype(np.float32)
    b = np.random.RandomState(1).randn(n).astype(np.float32)
    c = np.zeros(n, dtype=np.float32)
    res, h, p = _run(
        "vector_add",
        ROOT / "examples/vector_add/kernel.ptx",
        grid=(8,1,1), block=(128,1,1),
        params={"A": a, "B": b, "C": c, "N": n},
    )
    correct = bool(np.allclose(c, a + b, rtol=1e-5))
    return res, h, p, {"correct": correct}


def reduction_smem():
    rng = np.random.RandomState(0)
    a = rng.randint(-100, 100, size=32).astype(np.int32)
    out = np.zeros(1, dtype=np.int32)
    res, h, p = _run(
        "reduction_smem",
        ROOT / "examples/reduction_smem/kernel.ptx",
        grid=(1,1,1), block=(32,1,1),
        params={"A": a, "OUT": out},
    )
    correct = int(out[0]) == int(a.sum())
    return res, h, p, {"correct": correct, "sum": int(out[0]), "expected": int(a.sum())}


def tiled_matmul():
    rng = np.random.RandomState(1)
    A = rng.randn(16,16).astype(np.float32)
    B = rng.randn(16,16).astype(np.float32)
    C = np.zeros((16,16), dtype=np.float32)
    res, h, p = _run(
        "tiled_matmul",
        ROOT / "examples/tiled_matmul/kernel.ptx",
        grid=(1,1,1), block=(16,16,1),
        params={"A": A, "B": B, "C": C},
    )
    correct = bool(np.allclose(C, A @ B, rtol=1e-4, atol=1e-4))
    return res, h, p, {"correct": correct, "max_err": float(np.max(np.abs(C - A @ B)))}


def divergence_demo():
    out = np.zeros(32, dtype=np.uint32)
    res, h, p = _run(
        "divergence_demo",
        ROOT / "examples/divergence_demo/kernel.ptx",
        grid=(1,1,1), block=(32,1,1),
        params={"OUT": out},
    )
    expected = np.array([100 if i < 16 else 200 for i in range(32)], dtype=np.uint32)
    return res, h, p, {"correct": bool(np.array_equal(out, expected))}


def bank_conflict_demo():
    out = np.zeros(32, dtype=np.uint32)
    res, h, p = _run(
        "bank_conflict_demo",
        ROOT / "examples/bank_conflict_demo/kernel.ptx",
        grid=(1,1,1), block=(32,1,1),
        params={"OUT": out},
    )
    return res, h, p, {"correct": bool(np.array_equal(out, np.arange(32, dtype=np.uint32)))}


def coalescing_demo():
    n = 1024
    a = np.arange(n, dtype=np.uint32)
    results = {}
    last_res = last_h = last_p = None
    for stride in (1, 2, 4, 8):
        out = np.zeros(32, dtype=np.uint32)
        res, h, p = _run(
            f"coalescing_demo_s{stride}",
            ROOT / "examples/coalescing_demo/kernel.ptx",
            grid=(1,1,1), block=(32,1,1),
            params={"A": a, "OUT": out, "STRIDE": stride},
        )
        results[f"stride={stride}"] = {
            "cycles": res.metrics["cycles"],
            "first8": out[:8].tolist(),
        }
        last_res, last_h, last_p = res, h, p
    return last_res, last_h, last_p, results


CASES = [
    ("vector_add",        vector_add),
    ("reduction_smem",    reduction_smem),
    ("tiled_matmul",      tiled_matmul),
    ("divergence_demo",   divergence_demo),
    ("bank_conflict_demo",bank_conflict_demo),
    ("coalescing_demo",   coalescing_demo),
]


def main():
    print(f"writing reports + traces to {OUT}/")
    summary = {}
    for name, fn in CASES:
        try:
            res, h, p, extra = fn()
            row = {
                "summary": res.summary(),
                "metrics": res.metrics,
                "html":    str(h),
                "perfetto":str(p),
                **extra,
            }
            print(f"\n== {name} ==")
            for k, v in row.items():
                print(f"  {k}: {v}")
            summary[name] = row
        except Exception as e:
            print(f"\n== {name} == FAILED: {type(e).__name__}: {e}")
            summary[name] = {"error": f"{type(e).__name__}: {e}"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nfull summary: {OUT}/summary.json")


if __name__ == "__main__":
    sys.exit(main() or 0)
