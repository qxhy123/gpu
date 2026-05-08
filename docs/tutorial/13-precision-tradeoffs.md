# Chapter 13 — 精度面板

## 6 种精度：位宽与用途

Modern Tensor Core accelerators support multiple numeric formats, each trading range and precision for throughput or memory bandwidth. The simulator's `examples/tc_matmul_precisions/` demonstrates all six:

| Format | Total bits | Sign | Exponent | Mantissa | Typical use |
|--------|-----------|------|----------|----------|-------------|
| FP32   | 32        | 1    | 8        | 23       | Training accumulator, reference |
| FP16   | 16        | 1    | 5        | 10       | Training weights & activations |
| BF16   | 16        | 1    | 8        | 7        | Training (wide range), LLM inference |
| TF32   | 19        | 1    | 8        | 10       | Tensor Core drop-in for FP32 |
| FP8 E4M3 | 8      | 1    | 4        | 3        | Inference, low-precision fine-tuning |
| INT8   | 8         | 1    | —        | 7 (int)  | Inference, quantized models |

The PTX kernels in `examples/tc_matmul_precisions/` each use the corresponding `mma.sync` variant. The simulator runs all six and reports cycle counts and numeric error.

## 从模拟器看：对比表

```bash
python examples/tc_matmul_precisions/run.py
```

Typical output:

```
# tc_matmul_precisions: 6 dtype variants
variant  cycles   max diff vs numpy
fp32     1280     0.00e+00
fp16     420      3.81e-06
bf16     420      3.81e-06
e4m3     420      1.56e-02
tf32     420      9.54e-07
int8     420      0.00e+00
```

Reading this table:
- **Cycles**: FP32 (CUDA Core scalar FMA) is ~3× slower than all Tensor Core variants for this small 16×8 problem, because scalar FMA must issue many individual instructions.
- **Max diff**: INT8 has zero error because integer arithmetic is exact (no rounding). FP32 CUDA Core also shows zero error against the NumPy FP64 reference at this scale. FP16 has ~4 ULP error. E4M3 has the largest error (~0.016) due to its 3-bit mantissa.

## 为什么 FP8 误差大

FP8 E4M3 has only 3 bits of mantissa (significand), giving 2³ = 8 gradations between powers of two. For comparison, FP16 has 10 mantissa bits (1024 gradations). The representable step size (machine epsilon) is:
- FP16: ε ≈ 2^{-10} ≈ 0.001
- FP8 E4M3: ε ≈ 2^{-3} = 0.125

Beyond precision, E4M3 has only 4 exponent bits, giving a maximum representable value of 448 (versus 65504 for FP16). When inputs exceed this range, E4M3 **saturates** — values clamp to ±448 instead of going to ±Inf as FP16 would. Saturated inputs produce wildly incorrect dot products.

Run the following to see the saturation effect:

```python
import numpy as np
from examples.tc_matmul_precisions.reference import build_inputs, reference_output

# Default inputs (stddev ~1): works fine
A, B, C = build_inputs("e4m3", seed=0)
print("max |A|:", np.max(np.abs(A)))   # ~4.0, well within E4M3 range

# Scale inputs by 100: E4M3 saturates at 448
A_big = A * 100
print("max |A_big|:", np.max(np.abs(A_big)))  # ~400, close to limit
# After E4M3 quantization, most values will clamp to 448.0
```

## BF16 vs FP16：mantissa 7 vs 10

Both BF16 and FP16 use 16 bits total, but they distribute bits differently:

```
FP16:  [S 1][EEEEE 5][MMMMMMMMMM 10]   max value ≈ 65504
BF16:  [S 1][EEEEEEEE 8][MMMMMMM 7]    max value ≈ 3.4 × 10³⁸
```

BF16 matches FP32's 8-bit exponent exactly, so it can represent the same *range* of values as FP32. Converting FP32 weights to BF16 never causes overflow or NaN from out-of-range values — only rounding error (precision loss).

FP16, by contrast, has exponent range only up to ~65504. When FP32 weights have values like 70000.0, converting to FP16 produces ±Inf. This is a common source of training instability. BF16 training is more numerically stable for large models.

Precision comparison: FP16 has 10 mantissa bits (0.1% relative error per operation), BF16 has 7 (0.8% relative error). For matrix multiplications accumulating K terms, the error grows as `~sqrt(K) × eps`. For K=16 (our test):
- FP16: sqrt(16) × 0.001 ≈ 0.004
- BF16: sqrt(16) × 0.008 ≈ 0.032

The simulator confirms: both FP16 and BF16 show `max diff ≈ 3.8e-06` against the NumPy FP64 reference — negligible for K=16.

## TF32：FP32 输入 + 10-bit mantissa 截断

TF32 (TensorFloat-32) is an NVIDIA-internal format with 19 total bits:
- 1 sign bit
- 8 exponent bits (same as FP32)
- 10 mantissa bits (same as FP16)

It is **not an IEEE standard format**. The kernel `kernel_tf32.ptx` reads FP32 inputs directly, but the Tensor Core internally truncates the mantissa from 23 bits to 10 before computing. The output D is FP32.

This means:
- **No format conversion needed** in the kernel — inputs are stored as FP32.
- **Precision** is FP16-level (10 mantissa bits), not FP32-level (23 bits).
- **Range** matches FP32 — no saturation issues for typical values.

The mma variant is `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`. The simulator shows `max diff ≈ 9.5e-07`, slightly better than FP16 (because TF32's precision is comparable but the test inputs happen to round favorably).

## 改一改

**Experiment 1 — FP8 saturation:** Modify `reference.py` or `run.py` to scale A and B by 100× before converting to E4M3, then re-run. You should see the max diff spike dramatically (from ~0.016 to >1000) as the scaled values saturate at the E4M3 maximum of 448.0.

In `examples/tc_matmul_precisions/reference.py`, the `build_inputs` function generates inputs with `rng.randn(...)`, which has stddev~1. Multiplying by 100 pushes inputs to stddev~100:

```python
# In a modified run script:
A, B, C = build_inputs("e4m3", seed=0)
A = A * 100   # Force saturation
```

The expected result (numpy FP64) would be ~100× larger in magnitude, but E4M3-quantized A would have most values clamped to ±448.0. The output would be dominated by saturation error.

**Experiment 2 — INT8 range:** Unlike FP formats, INT8 saturates at 127/−128. Modify inputs so some element exceeds 127 and verify that the quantized kernel produces wrong results.

## 真机对照

On H100 SXM5 (from the datasheet):
- **FP16 Tensor Core**: 989 TFLOPS dense
- **BF16 Tensor Core**: 989 TFLOPS dense (same as FP16 — same bit width, same hardware path)
- **FP8 Tensor Core**: 1979 TFLOPS dense (**2× FP16**) — FP8 inputs pack two values per FP16 slot
- **INT8 Tensor Core**: 1979 TFLOPS dense (same as FP8 — same 8-bit width)
- **TF32 Tensor Core**: 494 TFLOPS dense (19-bit operands, half FP16 throughput)

The 2× FP8 advantage comes from packing: each 128-bit register holds 16 FP8 values vs. 8 FP16 values, so the hardware can feed twice as many operands per cycle.

In the simulator, cycle counts are similar across all Tensor Core variants because the bottleneck for this micro-kernel is the global memory loads, not the mma instruction. For compute-bound workloads (large K), the 2× FP8 advantage would manifest.
