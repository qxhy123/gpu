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
- Cache hierarchy (L1, L2, HBM) — `examples/l1_thrash_demo/` + Chapter 08
- Shared memory vs L1 cache — `examples/smem_vs_l1_demo/` + Chapter 09
- HBM bandwidth saturation — `examples/bw_saturation_demo/` + Chapter 10
- DRAM row buffer locality — `examples/row_buffer_demo/` + Chapter 11

## Run a kernel and inspect the report
```bash
# Option 1: Python API (no .npy files needed)
python -c "
import numpy as np, gpusim, pathlib
n = 1024
a = np.random.randn(n).astype(np.float32)
b = np.random.randn(n).astype(np.float32)
c = np.zeros(n, dtype=np.float32)
ptx = pathlib.Path('examples/vector_add/kernel.ptx').read_text()
res = gpusim.run(ptx_src=ptx, grid=(8,1,1), block=(128,1,1),
                 params={'A':a,'B':b,'C':c,'N':n}, mode='timing')
res.html_report('report.html')
res.perfetto('trace.json')
print(res.summary())
"

# Option 2: CLI (after staging .npy inputs)
python -c "import numpy as np; np.save('a.npy', np.random.randn(1024).astype(np.float32)); np.save('b.npy', np.random.randn(1024).astype(np.float32)); np.save('c.npy', np.zeros(1024, dtype=np.float32))"
gpusim run examples/vector_add/kernel.ptx \
    --grid 8 --block 128 \
    --inputs A:a.npy,B:b.npy,C:c.npy,N:1024 \
    --mode timing \
    --output report.html --perfetto trace.json
```
- `report.html` — open in any browser
- `trace.json` — drag into https://ui.perfetto.dev for an interactive timeline

## What's modeled (Phase 2)
Single SM, cycle-approximate, Hopper-shaped. PTX subset (~30 ops). Shared memory bank conflicts, global memory coalescing, regfile bank conflicts, multi-CTA occupancy. **Cache hierarchy: tag-precise L1 (128 KB / 4-way / 16 MSHR), L2 (4 MB / 16-way / write-back), HBM (8 channels × 16 banks + row buffer + channel queue).** Cache hit rates, HBM bandwidth utilization, row buffer locality, write-back traffic all surfaced in the HTML report and Python API.

## What's NOT modeled
Tensor Core, FP16/BF16/FP8, TMA, thread-block clusters, warp shuffle, ITS, multi-SM, multi-GPU. See `docs/superpowers/specs/2026-05-07-gpusim-phase1-design.md` section 11.

## Tutorial
Read `docs/tutorial/00-intro.md` first.
