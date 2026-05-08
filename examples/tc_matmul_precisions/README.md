# tc_matmul_precisions

6 PTX variants computing the **same** 16x8 matmul in different precisions to surface speed/accuracy trade-offs.

## Variants

| Variant | Dtype A/B | Dtype Accum | K | Tol vs FP32 ref |
|---|---|---|---|---|
| fp32 | f32 | f32 | 16 | 1e-5 (baseline; no Tensor Core) |
| fp16 | f16 | f32 | 16 | 1e-2 |
| bf16 | bf16 | f32 | 16 | 1e-2 |
| e4m3 | e4m3 | f32 | 32 | 2e-1 |
| tf32 | tf32 | f32 | 8 | 1e-3 |
| int8 | s8 | s32 | 32 | 0 (exact) |

## Key Points

- `fp32` does not use Tensor Core; each thread computes 4 output elements via scalar FMA loop -> cycles far higher than mma variants
- FP16/BF16 differ in mantissa length (FP16: 10-bit vs BF16: 7-bit) but similar accuracy at this scale
- FP8 (e4m3) shows significant precision loss (~10-20%) but processes 2x more K elements per mma
- TF32 is FP32 with truncated mantissa (10-bit precision), can handle FP32 inputs
- INT8 is exact (no rounding error in integer arithmetic)

## Running

```bash
python examples/tc_matmul_precisions/run.py
```

## Discussion Questions

1. How does FP8 error propagate through a full model? Hint: with N accumulation steps, error grows as ~sqrt(N) * eps
2. Why is INT8 exact while FP8 is not?
3. Which has a larger epsilon: TF32 or BF16?
