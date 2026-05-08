# Chapter 12 — Tensor Core 入门

## 什么是 Tensor Core（vs CUDA Core）

A **CUDA Core** is a scalar ALU: it takes two floating-point scalars and produces one result per cycle. A warp of 32 lanes can issue one `mul.f32` or `fma.f32` every cycle, so the peak throughput for a 32-lane warp is 32 multiplies per cycle.

A **Tensor Core** operates on entire matrices in one instruction. The PTX instruction `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` computes:

```
D[16×8] = A[16×16] × B[16×8] + C[16×8]
```

in a single *warp-cooperative* operation. All 32 lanes of the warp collaborate: each lane holds a portion of A, B, and C in its registers, and after the instruction executes, each lane holds the corresponding portion of D. The hardware dispatches the multiply-accumulate across the Tensor Core matrix engines in the background.

For an m16n8k16 FP16 tile, the total FLOPs = 2 × 16 × 8 × 16 = 4096. A CUDA Core path performing the same operation via 4096 scalar FMAs in one warp would take ~4096 / 32 = 128 cycles. The Tensor Core path takes just a few cycles — roughly 16–32× throughput gain for the compute portion.

## sync mma 指令格式

The PTX mma instruction follows the form:

```ptx
mma.sync.aligned.m{M}n{N}k{K}.row.col.{Dtype}.{AType}.{BType}.{CType}
    {D regs}, {A regs}, {B regs}, {C regs};
```

Key fields:
- `sync` — the entire warp executes together; no lane may diverge during the instruction.
- `aligned` — the register layout must match the documented lane-to-element mapping.
- `m{M}n{N}k{K}` — tile dimensions. The most common for FP16 is `m16n8k16`.
- `row.col` — A is row-major, B is column-major in the register layout.
- Dtype / AType / BType / CType — output, A input, B input, C accumulator types.

For `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`:
- A input: FP16, 8 registers per lane (8 fp16 values)
- B input: FP16, 4 registers per lane
- C/D accumulator: FP32, 4 registers per lane

## m16n8k16 FP16 形状解析

The conceptual matrix sizes are A[16×16], B[16×8], D[16×8]. But these 16×16, 16×8, 16×8 matrices are distributed across all 32 lanes. Each lane holds only a slice:

```
For lane L (0..31):
  row     = L / 2
  col_half = L % 2

  A registers %a0..%a7:
    A[row][col_half*8 .. col_half*8+7]   (8 consecutive FP16 in row-major order)

  B registers %b0..%b3:
    B[row][col_half*4 .. col_half*4+3]   (4 consecutive FP16 in col-major order)

  D registers %d0..%d3:
    D[row][col_half*4 .. col_half*4+3]   (4 FP32 results)
```

Lanes 0 and 1 both cover row 0 of A, but lane 0 covers columns 0–7 and lane 1 covers columns 8–15. Together they cover the full A row. After `mma.sync`, lane 0's %d0..%d3 hold D[0][0..3] and lane 1's %d0..%d3 hold D[0][4..7].

## 走通 kernel_fp16.ptx

Open `examples/tc_matmul_precisions/kernel_fp16.ptx`. The kernel has three phases:

**Phase 1 — Load A (lines 27–47):**
```ptx
shr.u32 %r1, %r0, 1;       // row = lane / 2
and.b32 %r2, %r0, 1;       // col_half = lane % 2
// byte offset for A = (row*16 + col_half*8) * 2
// loads 8 fp16 into %a0..%a7
ld.global.f16 %a0, [%rd5];
...
ld.global.f16 %a7, [%rd5 + 14];
```

Each lane loads 8 consecutive FP16 values from row `lane/2` of matrix A. The stride `lane/2` ensures that lanes 0 and 1 (same row) load adjacent 8-element blocks.

**Phase 2 — Load B (lines 49–59):**
```ptx
// byte offset for B = (row*8 + col_half*4) * 2
ld.global.f16 %b0, [%rd7];
...
ld.global.f16 %b3, [%rd7 + 6];
```

Each lane loads 4 FP16 values from B.

**Phase 3 — mma.sync + store D (lines 68–85):**
```ptx
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {%d0, %d1, %d2, %d3},
    {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7},
    {%b0, %b1, %b2, %b3},
    {%c0, %c1, %c2, %c3};
// then store %d0..%d3 back to OUT
```

The C accumulator is zero-initialized (`mov.f32 %c0, 0.0`), so D = A×B with no bias addition.

## 看模拟器

Run the full precision comparison to see FP16 vs FP32 CUDA Core baseline:

```bash
python examples/tc_matmul_precisions/run.py
```

Expected output (cycle counts vary with simulator version):

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

Key observations:
1. `fp32` uses scalar FMA — cycles are ~3× more than the Tensor Core variants. This is the CUDA Core baseline.
2. All Tensor Core variants (`fp16`, `bf16`, `e4m3`, `tf32`, `int8`) run in roughly the same cycle count — the bottleneck is memory loads, not the mma instruction itself for this small matrix.
3. `fp16` has tiny floating-point rounding error (`3.81e-06`) vs. numpy's FP64 reference — expected for FP16 arithmetic.

In `report.html`, look at the **instruction mix** section. The fp16 kernel has exactly one `mma.sync` instruction. The fp32 kernel has a long sequence of `fma.f32` instructions. The Tensor Core instruction count is dramatically lower.

## 改一改

The `m16n8k16` shape uses K=16 (16 columns of A, 16 rows of B). There is also an `m16n8k8` shape for FP16:

```ptx
// Change kernel to m16n8k8:
// A[16×8] — each lane loads 4 fp16 (half as many)
// B[8×8]  — each lane loads 2 fp16
mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32
    {%d0, %d1, %d2, %d3},
    {%a0, %a1, %a2, %a3},
    {%b0, %b1},
    {%c0, %c1, %c2, %c3};
```

With K=8, each `mma.sync` computes half as much work. To compute the same full 16×8 result from a 16×16 × 16×8 product, you would need to split K=16 into two K=8 tiles and accumulate. The two mma instructions have the same total cycle budget, but K=8 may be useful when the K dimension is small (e.g., the first tile of a K=8 GeMM).

Try modifying `kernel_fp16.ptx` to use m16n8k8 with a 16×8 A and 8×8 B matrix and verify the output.

## 真机対照

On a real H100 SXM5, the Tensor Core throughput for FP16 (m16n8k16) is listed in the datasheet as:
- **Tensor Core FP16**: 1979 TFLOPS (sparse) / 989 TFLOPS (dense)
- **CUDA Core FP32**: 66.9 TFLOPS

The ratio is ~15×, consistent with the simulator's ~3× cycle difference for this micro-kernel (memory-bound here, not compute-bound). For a compute-bound kernel with large K, the gap widens toward the full 15–16× advantage.

The simulator reproduces the *qualitative* ratio: one `mma.sync` produces the same output as many `fma.f32` instructions, in far fewer cycles.
