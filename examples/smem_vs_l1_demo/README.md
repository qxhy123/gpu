# smem_vs_l1_demo

Same 16x16 matmul in two variants:
- `kernel_smem.ptx`: manual shared-memory tiling (alias of Phase 1 tiled_matmul)
- `kernel_no_smem.ptx`: no smem, relies on L1 cache for reuse

## Run

```
python examples/smem_vs_l1_demo/run.py
```

## Expected observations (Phase 2 timing mode)

- Both variants produce numerically identical results
- `kernel_smem.ptx`: HBM traffic = one-time input load (~32 cache lines); fewer total cycles
- `kernel_no_smem.ptx`: similar HBM traffic (L1 captures reuse), but significantly more L1 lookups
- HTML report §6 cache hit rate: smem variant ~0% L1 (bypasses via shared mem); no_smem variant ≥95% L1 hit

## Discussion

- "L1 cache captures reuse vs manual smem captures reuse": which wins?
- Capacity comparison: 256 KB SRAM fully for smem requires warp coordination; fully for L1 uses LRU automatically
- Control vs automation tradeoff

## Extension ideas

1. Scale matmul to 64x64 — can no_smem still capture all reuse in default_hopper.yaml L1=128KB?
2. Set `l1_size_bytes: 4096` (4 KB tiny L1) and rerun: no_smem performance degrades, smem variant stays stable
