# Chapter 14 — 混合精度 Accumulator

## 为什么 accumulator 用 FP32 而不是 FP16

When a Tensor Core instruction computes D = A×B + C, the *accumulator* — the C and D registers — can be either FP16 or FP32, even when A and B are FP16. Why does it matter?

Each mma instruction accumulates K multiply-add operations. For a full matmul with K=1024 (a common inner-loop dimension in transformer models), each output element D[i,j] is the sum of 1024 products `A[i,k] × B[k,j]`. With FP16 accumulation, every addition rounds to the nearest FP16 value (10 mantissa bits ≈ 0.1% relative error). With FP32 accumulation, the same additions round to the nearest FP32 value (23 mantissa bits ≈ 1.2×10⁻⁷ relative error).

The error compounds with each addition. For K summands accumulated in FP16:

```
Accumulated error ≈ sqrt(K) × ε_fp16
                  ≈ sqrt(K) × 2^{-10}
```

For K=64: `sqrt(64) × 0.001 = 0.008` — about 0.8% relative error.
For K=1024: `sqrt(1024) × 0.001 = 0.032` — about 3.2% relative error on every output element.

With FP32 accumulation at K=1024: `sqrt(1024) × 1.2×10⁻⁷ ≈ 3.8×10⁻⁶` — 8000× more accurate.

For model training, FP16 accum error at K=1024 can push gradient magnitudes out of the range where FP16 can distinguish them, causing gradient collapse or NaN propagation.

## 从 examples/mixed_accum 看

The `examples/mixed_accum/` example runs two kernels: `kernel_fp32_accum.ptx` and `kernel_fp16_accum.ptx`. Both compute the same 16×8 matmul accumulated over 64 K-tiles (K_ITERS=64), producing a 16×8 result. The difference is only the D register type.

Run it:

```bash
python examples/mixed_accum/run.py
```

Expected output:

```
# mixed_accum: FP16 vs FP32 accumulator (64 K-tile iterations)
variant        cycles     max diff vs fp32 ref
fp32_accum     6820       8.3923e-05
fp16_accum     6820       8.5449e-03
```

Key observations:
1. **Cycle counts are equal** — the accumulator type does not change the mma instruction's compute throughput. FP16 and FP32 accumulators take the same cycles in the Tensor Core.
2. **Error difference is ~100×** — FP32 accum shows max diff ~8e-5; FP16 accum shows ~8.5e-3. The FP16 error is larger by exactly the expected factor: sqrt(64) × (ε_fp16/ε_fp32) ≈ 8 × 8000 ≈ 65000, tempered by the fact that errors partially cancel (random walk vs. worst case).

The `max diff` is measured against a reference computed in FP32 (numpy with `A.astype(float32) @ B.astype(float32)`), which itself has FP32 rounding error. The FP32 accum kernel matches this reference closely; the FP16 accum kernel has ~100× more error.

## 数学背景：累加误差分析

For a dot product of K terms, each with magnitude ~1:

```
D = Σ_{k=0}^{K-1} a_k × b_k
```

Each multiply `a_k × b_k` has rounding error ε. Each addition has rounding error ε. After K additions:
- **Worst case** (all errors same sign): total error = K × ε
- **Expected case** (errors are independent, zero-mean): total error = sqrt(K) × ε (random walk)

For FP16 (ε ≈ 2^{-10} ≈ 0.001) and K=64:
- Expected error per output element: sqrt(64) × 0.001 = 0.008

For FP32 (ε ≈ 2^{-23} ≈ 1.2×10^{-7}) and K=64:
- Expected error per output element: sqrt(64) × 1.2×10^{-7} = 9.5×10^{-7}

The ratio: 0.008 / 9.5×10^{-7} ≈ 8400, consistent with the ~100× we see in the simulator (the simulator uses K=64 tiles but each tile's contribution is bounded, not unbounded random walk).

## 看模拟器：kernel 差异

Open `examples/mixed_accum/kernel_fp32_accum.ptx`. The mma variant is:

```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%d0, %d1, %d2, %d3}, ...
```

The last `.f32` specifies that D is FP32. The 64 K-tile loop accumulates into `%d0..%d3` (FP32 registers). Each iteration: C = D (previous result) is passed back in as the new C, accumulating.

Open `examples/mixed_accum/kernel_fp16_accum.ptx`. The mma variant is:

```ptx
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
    {%d0, %d1}, ...
```

Here D and C are FP16. After each K-tile, the FP16 result is rounded back to 10 mantissa bits before the next addition — that is the rounding error accumulating over 64 iterations.

## 改一改

**Increase K_ITERS to 128:**

Modify `examples/mixed_accum/run.py` line:
```python
params={"A": A, "B": B, "OUT": out, "K_ITERS": 64},
```
to:
```python
params={"A": A, "B": B, "OUT": out, "K_ITERS": 128},
```

Also change `A_full` and `B_full` to use K=16×128=2048:
```python
A_full = rng.randn(16, 16 * 128).astype(np.float16)
B_full = rng.randn(16 * 128, 8).astype(np.float16)
```

With K_ITERS=128, the expected FP16 accum error should grow by roughly sqrt(128/64) = sqrt(2) ≈ 1.41×. If fp16_accum was 8.5e-3, it should reach ~0.012. The fp32_accum error grows negligibly. The ratio grows to ~140×.

At K_ITERS=256 (total K=4096), FP16 accum error becomes large enough to visually corrupt the output in some elements — the ~3% relative error per element means output values could be wrong by ~1–10 depending on magnitude.

**Verify without accumulation (K_ITERS=1):**

With `K_ITERS=1`, both variants should produce nearly identical results (only one mma.sync, no accumulated rounding). This confirms the error is entirely due to accumulation, not the initial mma computation.

## 真机对照

In cuBLAS and CUTLASS, the default accumulator for FP16 matmul is **FP32**, regardless of input dtype:

```c++
// cuBLAS FP16 GeMM: default compute type is CUBLAS_COMPUTE_32F
cublasGemmEx(handle, ...,
    CUDA_R_16F, CUDA_R_16F, CUDA_R_16F,  // A, B, C types
    CUBLAS_COMPUTE_32F,                   // accumulator = FP32
    ...);
```

CUTLASS similarly defaults to `ElementAccumulator = float` for all FP16 kernels. Using FP16 accumulation requires explicitly opting in (e.g., CUTLASS's `ElementAccumulator = half_t`), and it is generally not recommended for training workloads.

The H100 Tensor Core hardware supports both FP16 and FP32 accumulators for FP16 inputs with no throughput penalty — the choice is entirely about numerical precision.
