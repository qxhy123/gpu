import numpy as np
from gpusim.core.tensor_core.precision import (
    numpy_dtype_for, storage_bytes, cast_array, cast_scalar,
)
from gpusim.frontend.ir import PtxType


def test_numpy_dtype_f32():
    assert numpy_dtype_for(PtxType.f32) == np.float32

def test_numpy_dtype_f16():
    assert numpy_dtype_for(PtxType.f16) == np.float16

def test_numpy_dtype_bf16():
    import ml_dtypes
    assert numpy_dtype_for(PtxType.bf16) == ml_dtypes.bfloat16

def test_numpy_dtype_e4m3():
    import ml_dtypes
    assert numpy_dtype_for(PtxType.e4m3) == ml_dtypes.float8_e4m3fn

def test_numpy_dtype_e5m2():
    import ml_dtypes
    assert numpy_dtype_for(PtxType.e5m2) == ml_dtypes.float8_e5m2

def test_numpy_dtype_int8():
    assert numpy_dtype_for(PtxType.s8) == np.int8

def test_numpy_dtype_tf32_returns_float32():
    # TF32 stored as float32; truncation handled at cast time
    assert numpy_dtype_for(PtxType.tf32) == np.float32

def test_storage_bytes():
    assert storage_bytes(PtxType.f32) == 4
    assert storage_bytes(PtxType.f16) == 2
    assert storage_bytes(PtxType.bf16) == 2
    assert storage_bytes(PtxType.e4m3) == 1
    assert storage_bytes(PtxType.e5m2) == 1
    assert storage_bytes(PtxType.s8) == 1
    assert storage_bytes(PtxType.tf32) == 4   # stored as f32

def test_cast_round_trip_f16():
    a = np.array([1.0, 2.5, -0.125], dtype=np.float32)
    b = cast_array(a, src=PtxType.f32, dst=PtxType.f16)
    assert b.dtype == np.float16
    c = cast_array(b, src=PtxType.f16, dst=PtxType.f32)
    assert np.allclose(a, c, atol=1e-3)

def test_cast_tf32_truncates_mantissa():
    # TF32 uses 10-bit mantissa; values should round to nearest representable.
    # tf32 stored as float32, but cast must truncate mantissa.
    a = np.array([1.0 + 1e-7], dtype=np.float32)
    b = cast_array(a, src=PtxType.f32, dst=PtxType.tf32)
    # tf32 mantissa precision ~ 2^-10 ≈ 1e-3 → 1e-7 lost
    assert b.dtype == np.float32
    assert b[0] == 1.0  # tf32 rounds 1.0+1e-7 down to 1.0

def test_cast_scalar():
    v = cast_scalar(1.5, src=PtxType.f32, dst=PtxType.f16)
    assert isinstance(v, float) or hasattr(v, "dtype")
    assert abs(float(v) - 1.5) < 1e-3
