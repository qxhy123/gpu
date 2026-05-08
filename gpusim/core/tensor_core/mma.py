"""Sync mma functional execution. Lane-to-element layout is fictional/simplified
(spec §11) — does NOT match real PTX register-to-element mapping. Numerically
correct via numpy + ml_dtypes."""
from __future__ import annotations
import numpy as np
from gpusim.core.exec import WarpFnState
from gpusim.frontend.ir import RegGroup, PtxType
from gpusim.core.tensor_core.mma_spec import MmaSpec
from gpusim.core.tensor_core.precision import numpy_dtype_for, cast_array


def _collect_a(w: WarpFnState, group: RegGroup, M: int, K: int,
               dtype: PtxType) -> np.ndarray:
    """Read A[M][K]. Layout: lane i, reg j -> A[i/2][(i%2) * (K/2) + j]."""
    half_K = K // 2
    out = np.zeros((M, K), dtype=np.float32)
    n_regs = len(group.regs)
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * half_K
        for j in range(n_regs):
            out[row, col_base + j] = w.threads[lane].get_f32(group.regs[j].name)
    return cast_array(out, src=PtxType.f32, dst=dtype)


def _collect_b(w: WarpFnState, group: RegGroup, K: int, N: int,
               dtype: PtxType) -> np.ndarray:
    """Read B[K][N].
    For K=8 or K=16: 32 lanes cover 16 rows × N cols, 2 lanes/row.
        For K=16: lane i, reg j -> B[i/2][(i%2) * (N/2) + j], n_regs = N/2.
        For K=8 (TF32): rows_factor = 32 // K = 4, so 4 lanes per row, n_regs = N / rows_factor = 2.
    For K=32: 32 lanes cover all 32 rows, each lane handles N cols.
        Layout: lane i, reg j -> B[i][j]"""
    out = np.zeros((K, N), dtype=np.float32)
    n_regs = len(group.regs)
    if K <= 16:
        rows_factor = 32 // K
        cols_per_block = N // rows_factor
        for lane in range(32):
            row = lane // rows_factor
            col_base = (lane % rows_factor) * cols_per_block
            for j in range(n_regs):
                out[row, col_base + j] = w.threads[lane].get_f32(group.regs[j].name)
    else:
        # K=32: lane i covers row i, reg j covers col j
        for lane in range(32):
            for j in range(n_regs):
                out[lane, j] = w.threads[lane].get_f32(group.regs[j].name)
    return cast_array(out, src=PtxType.f32, dst=dtype)


def _collect_d_or_c(w: WarpFnState, group: RegGroup, M: int, N: int,
                     dtype: PtxType) -> np.ndarray:
    """Read D/C[M][N]. Layout: lane i, reg j -> D[i/2][(i%2)*(N/2) + j]."""
    half_N = N // 2
    out = np.zeros((M, N), dtype=np.float32)
    n_regs = len(group.regs)
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * half_N
        for j in range(n_regs):
            out[row, col_base + j] = w.threads[lane].get_f32(group.regs[j].name)
    return cast_array(out, src=PtxType.f32, dst=dtype)


def _distribute_d(w: WarpFnState, group: RegGroup, M: int, N: int,
                   D: np.ndarray) -> None:
    """Write D[M][N] back into lane registers (D layout matches C).
    Integer dtypes (int32) are stored in both s32 and f32 slots for cross-type reads."""
    half_N = N // 2
    n_regs = len(group.regs)
    is_int = D.dtype in (np.int32, np.int16, np.int8)
    D32 = D.astype(np.float32)
    for lane in range(32):
        row = lane // 2
        col_base = (lane % 2) * half_N
        for j in range(n_regs):
            val = D32[row, col_base + j]
            w.threads[lane].set_f32(group.regs[j].name, float(val))
            if is_int:
                w.threads[lane].set_s32(group.regs[j].name, int(val))
                w.threads[lane].set_u32(group.regs[j].name, int(val) & 0xFFFFFFFF)


def execute_mma(spec: MmaSpec, w: WarpFnState,
                 dst: RegGroup, a: RegGroup, b: RegGroup, c: RegGroup) -> None:
    """sync mma: D = A @ B + C. Functional only; no timing."""
    A = _collect_a(w, a, spec.m, spec.k, spec.dtype_a)
    B = _collect_b(w, b, spec.k, spec.n, spec.dtype_b)
    C = _collect_d_or_c(w, c, spec.m, spec.n, spec.dtype_c)
    D = (A.astype(np.float32) @ B.astype(np.float32) + C.astype(np.float32))
    D_typed = cast_array(D, src=PtxType.f32, dst=spec.dtype_d)
    _distribute_d(w, dst, spec.m, spec.n, D_typed)
