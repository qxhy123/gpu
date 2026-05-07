# gpusim

Teaching-oriented NVIDIA GPU microarchitecture simulator.

## Quick start
```bash
pip install -e ".[dev]"
gpusim doctor
python examples/vector_add/run.py
```

## What you can learn
- SIMT execution + branch divergence (`examples/divergence_demo/`)
- Global memory coalescing (`examples/coalescing_demo/`)
- Shared memory bank conflicts (`examples/bank_conflict_demo/`)
- Reduction with shared memory + bar.sync (`examples/reduction_smem/`)
- Tiled matmul (`examples/tiled_matmul/`)
- Occupancy bottlenecks (`docs/tutorial/06-occupancy.md`)

## Run a kernel and inspect the report
```bash
gpusim run examples/vector_add/kernel.ptx \
    --grid 8 --block 128 \
    --inputs A:a.npy,B:b.npy,C:c.npy,N:1024 \
    --output report.html --perfetto trace.json
```
- `report.html` — open in any browser
- `trace.json` — drag into https://ui.perfetto.dev for an interactive timeline

## What's modeled (Phase 1)
Single SM, cycle-approximate, Hopper-shaped. PTX subset (~30 ops). Shared memory bank conflicts, global memory coalescing, regfile bank conflicts, multi-CTA occupancy.

## What's NOT modeled (Phase 1)
Tensor Core, FP16/BF16/FP8, L1/L2 cache, HBM bandwidth, TMA, thread-block clusters, warp shuffle, ITS, multi-SM, multi-GPU. See `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` section 11.

## Tutorial
Read `docs/tutorial/00-intro.md` first.
