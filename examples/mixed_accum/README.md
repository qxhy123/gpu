# mixed_accum

Demonstrates the precision impact of FP16 vs FP32 matrix-multiply accumulators across 64 K-tile iterations.

## Problem

Both kernels compute the same **16×(16×64) @ (16×64)×8** matmul decomposed into 64 K-tiles of K=16 each, using `mma.sync.aligned.m16n8k16` Tensor Core instructions. The only difference is the accumulator type:

| Variant | mma opcode | Accumulator | Output | Typical max diff |
|---|---|---|---|---|
| `fp32_accum` | `f32.f16.f16.f32` | FP32 | FP32 | < 5e-2 |
| `fp16_accum` | `f16.f16.f16.f16` | FP16 | FP16 | > 5e-2 |

## Why FP32 Accumulator Matters

FP16 has a **10-bit mantissa** (~3 decimal digits of precision). When 64 partial products are accumulated in FP16, rounding errors compound at each step. With values of moderate magnitude, the per-tile rounding error (~epsilon_fp16 ≈ 9.8e-4) accumulates over 64 iterations, resulting in a maximum difference of several tenths compared to the exact FP32 reference.

FP32 has a **23-bit mantissa** (~7 decimal digits). Accumulating 64 K-tiles in FP32 keeps rounding error well below 5e-2 even for random inputs.

In real GPU code (cuBLAS, Flash Attention, etc.) FP16 matmuls always accumulate in FP32 for this reason — the Tensor Core computes A×B products in reduced precision, but the running sum is kept in FP32 until the final store.

## Address Arithmetic

For each warp lane `tid` (0..31):
- `row = tid / 2`, `col_half = tid % 2`
- A lane base (fp16): `byte_offset = (row * K_total + col_half * 8) * 2`
- B lane base (fp16): `byte_offset = (row * 8 + col_half * 4) * 2`
- Per K-tile advance: A moves **+32 bytes** (16 K-cols × 2 bytes), B moves **+256 bytes** (16 K-rows × 8 N-cols × 2 bytes)

## Running

```bash
python examples/mixed_accum/run.py
```

Expected output (cycles depend on simulator config):
```
# mixed_accum: FP16 vs FP32 accumulator (64 K-tile iterations)
variant        cycles     max diff vs fp32 ref
fp32_accum     ...        <5e-02
fp16_accum     ...        >5e-02
```

## Parity Tests

```bash
.venv/bin/pytest tests/parity/test_mixed_accum.py -v
```

- `test_fp32_accum_preserves_precision` — max diff < 5e-2 (FP32 accum is accurate)
- `test_fp16_accum_loses_precision` — max diff > 5e-2 (FP16 accum diverges)

## Further Reading

See Tutorial 14 in the GPU Simulator documentation for a detailed walkthrough of Tensor Core accumulator precision and how this affects model training.
