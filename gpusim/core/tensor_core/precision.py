from __future__ import annotations
import numpy as np
import ml_dtypes
from gpusim.frontend.ir import PtxType


_NUMPY_DTYPE: dict[PtxType, np.dtype] = {
    PtxType.f32:  np.dtype(np.float32),
    PtxType.f16:  np.dtype(np.float16),
    PtxType.bf16: np.dtype(ml_dtypes.bfloat16),
    PtxType.e4m3: np.dtype(ml_dtypes.float8_e4m3fn),
    PtxType.e5m2: np.dtype(ml_dtypes.float8_e5m2),
    PtxType.tf32: np.dtype(np.float32),     # TF32 stored as float32; truncate at cast
    PtxType.s32:  np.dtype(np.int32),
    PtxType.s16:  np.dtype(np.int16),
    PtxType.s8:   np.dtype(np.int8),
    PtxType.u8:   np.dtype(np.uint8),
    PtxType.u32:  np.dtype(np.uint32),
}


def numpy_dtype_for(ty: PtxType) -> np.dtype:
    return _NUMPY_DTYPE[ty]


def storage_bytes(ty: PtxType) -> int:
    return _NUMPY_DTYPE[ty].itemsize


def _truncate_to_tf32(arr: np.ndarray) -> np.ndarray:
    """TF32 has 10-bit mantissa (vs FP32's 23-bit). Mask out the low 13 bits of the
    mantissa to simulate the precision loss."""
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)
    bits = arr.view(np.uint32).copy()
    bits &= 0xFFFFE000   # clear low 13 mantissa bits
    return bits.view(np.float32).copy()


def cast_array(arr: np.ndarray, *, src: PtxType, dst: PtxType) -> np.ndarray:
    if dst == PtxType.tf32:
        # TF32 = float32 with truncated mantissa
        if src == PtxType.tf32:
            return arr.copy()
        f32 = arr.astype(np.float32)
        return _truncate_to_tf32(f32)
    target = _NUMPY_DTYPE[dst]
    return arr.astype(target)


def cast_scalar(v: float | int, *, src: PtxType, dst: PtxType) -> float | int:
    arr = np.asarray([v], dtype=_NUMPY_DTYPE[src])
    out = cast_array(arr, src=src, dst=dst)
    return out.item() if out.dtype.kind in "fc" else int(out[0])
