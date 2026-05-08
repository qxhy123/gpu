# wgmma_basic

Minimal Hopper wgmma example: a single `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16`
instruction computed by one warp-group (128 threads / 4 warps).

## Problem

Multiply A (64×16 FP16) × B (16×128 FP16) → D (64×128 FP32).

## Kernel structure

1. **128 threads** (4 warps = 1 warp-group) launched per block.
2. Each thread copies 8 FP16 elements of A from gmem → shared memory (smem offset 0).
3. Each thread copies 16 FP16 elements of B from gmem → shared memory (smem offset 2048).
4. `bar.sync 0` — all threads synchronize.
5. All 4 warps cooperate on one `wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16`.
6. `wgmma.commit_group` + `wgmma.wait_group 0` — wait for completion.
7. Each thread writes its 64 FP32 D-registers back to the output buffer.

## Output layout

For warp `w` (0–3), lane `i` (0–31), register `j` (0–63):

```
D[w*16 + i/2][(i%2)*64 + j]
```

Each thread holds 64 FP32 values covering a 1×64 strip of the 64×128 output.

## Shared memory layout

| Region  | smem offset | Size (bytes) | Description       |
|---------|-------------|--------------|-------------------|
| smem_A  | 0           | 2048         | 64×16 FP16 matrix |
| smem_B  | 2048        | 4096         | 16×128 FP16 matrix|

## Running

```bash
python run.py
```

Expected output:
```
wgmma_basic: A=(64, 16) B=(16, 128) -> D=(64, 128)
  max |diff| = 0.000000   PASS
```

## Files

- `kernel.ptx` — PTX kernel
- `reference.py` — NumPy reference implementation
- `run.py` — driver script
