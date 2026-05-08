from gpusim.core.tensor_core.mma_spec import parse_mma_op, MmaSpec
from gpusim.frontend.ir import PtxType


def test_parse_sync_mma_fp16_k16():
    s = parse_mma_op("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32")
    assert s is not None
    assert s.is_async is False
    assert (s.m, s.n, s.k) == (16, 8, 16)
    assert s.layout_a == "row" and s.layout_b == "col"
    assert s.dtype_d is PtxType.f32
    assert s.dtype_a is PtxType.f16
    assert s.dtype_b is PtxType.f16
    assert s.dtype_c is PtxType.f32


def test_parse_sync_mma_bf16_k16():
    s = parse_mma_op("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
    assert s.dtype_a is PtxType.bf16


def test_parse_sync_mma_fp8_k32():
    s = parse_mma_op("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32")
    assert (s.m, s.n, s.k) == (16, 8, 32)
    assert s.dtype_a is PtxType.e4m3


def test_parse_sync_mma_tf32_k8():
    s = parse_mma_op("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32")
    assert (s.m, s.n, s.k) == (16, 8, 8)
    assert s.dtype_a is PtxType.tf32


def test_parse_sync_mma_int8_k32():
    s = parse_mma_op("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32")
    assert s.dtype_a is PtxType.s8
    assert s.dtype_d is PtxType.s32


def test_parse_wgmma_fp16():
    s = parse_mma_op("wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16")
    assert s.is_async is True
    assert (s.m, s.n, s.k) == (64, 128, 16)
    assert s.dtype_a is PtxType.f16
    # wgmma: dtype_c defaults to dtype_d if not in op string
    assert s.dtype_c is PtxType.f32


def test_parse_non_mma_returns_none():
    assert parse_mma_op("ld.global.f32") is None
    assert parse_mma_op("add.f32") is None
    assert parse_mma_op("wgmma.commit_group.sync.aligned") is None
