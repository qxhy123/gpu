import numpy as np
import ml_dtypes
from gpusim.frontend.ir import PtxType
from gpusim.core.tensor_core.precision import numpy_dtype_for, cast_array


_DTYPE = {
    "fp32":  np.float32,
    "fp16":  np.float16,
    "bf16":  ml_dtypes.bfloat16,
    "e4m3":  ml_dtypes.float8_e4m3fn,
    "tf32":  np.float32,
    "int8":  np.int8,
}

_PTX_TYPE = {
    "fp32": PtxType.f32, "fp16": PtxType.f16, "bf16": PtxType.bf16,
    "e4m3": PtxType.e4m3, "tf32": PtxType.tf32, "int8": PtxType.s8,
}


def output_dtype(variant: str):
    if variant == "int8":
        return np.int32
    return np.float32   # all float variants accumulate to f32


def build_inputs(variant: str, seed: int = 0):
    rng = np.random.RandomState(seed)
    K = {"fp32": 16, "fp16": 16, "bf16": 16, "e4m3": 32, "tf32": 8, "int8": 32}[variant]
    if variant == "int8":
        A = rng.randint(-8, 8, size=(16, K), dtype=np.int8)
        B = rng.randint(-8, 8, size=(K, 8), dtype=np.int8)
        C = np.zeros((16, 8), dtype=np.int32)
    else:
        A_f32 = rng.randn(16, K).astype(np.float32) * 0.5
        B_f32 = rng.randn(K, 8).astype(np.float32) * 0.5
        ty = _PTX_TYPE[variant]
        A = cast_array(A_f32, src=PtxType.f32, dst=ty)
        B = cast_array(B_f32, src=PtxType.f32, dst=ty)
        C = np.zeros((16, 8), dtype=np.float32)
    return A, B, C


def reference_output(A, B, C, variant: str):
    if variant == "int8":
        return (A.astype(np.int32) @ B.astype(np.int32)) + C
    return (A.astype(np.float32) @ B.astype(np.float32)) + C.astype(np.float32)
