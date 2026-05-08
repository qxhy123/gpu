from gpusim.frontend.ir import (
    Operand, Reg, Imm, Param, MemSpace, PtxType, Predicate,
    Instr, Kernel, RegDecl, SrcLoc,
)

def test_reg_operand_str_round_trip():
    op = Reg(name="r1", type=PtxType.s32)
    assert op.name == "r1"
    assert op.type is PtxType.s32

def test_imm_operand_value():
    op = Imm(value=42, type=PtxType.s32)
    assert op.value == 42

def test_predicate_negation():
    p = Predicate(reg="p1", negated=False)
    assert p.reg == "p1" and p.negated is False
    pn = Predicate(reg="p1", negated=True)
    assert pn.negated is True

def test_instr_immutable():
    instr = Instr(
        op="add.s32",
        dst=(Reg("r1", PtxType.s32),),
        src=(Reg("r2", PtxType.s32), Reg("r3", PtxType.s32)),
        pred=None,
        space=None,
        type=PtxType.s32,
        pc=0,
        src_loc=SrcLoc("k.ptx", 10),
    )
    import dataclasses
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        instr.op = "sub.s32"  # type: ignore[misc]

def test_kernel_holds_instr_list_and_labels():
    from types import MappingProxyType
    k = Kernel(
        name="vec_add",
        params=(Param(name="A", type=PtxType.b64),),
        regs=RegDecl(s32=4, f32=4, pred=2, b64=2),
        instrs=(),
        labels=MappingProxyType({"L1": 0}),
        ipdom=MappingProxyType({}),
    )
    assert k.name == "vec_add"
    assert k.labels["L1"] == 0
    assert k.regs.s32 == 4

def test_kernel_labels_immutable():
    from gpusim.frontend.parser import parse
    import pytest
    k = parse(".visible .entry foo() { .reg .u32 %r<1>; LBL: bar.sync 0; }", "<t>")
    with pytest.raises(TypeError):
        k.labels["new"] = 99  # type: ignore[index]

def test_phase3_ptx_types():
    from gpusim.frontend.ir import PtxType
    for t in ("f16", "bf16", "e4m3", "e5m2", "tf32", "s8", "u8", "s16"):
        assert PtxType(t).value == t

def test_ml_dtypes_importable():
    import ml_dtypes
    import numpy as np
    assert np.dtype(ml_dtypes.bfloat16).itemsize == 2
    assert np.dtype(ml_dtypes.float8_e4m3fn).itemsize == 1

def test_reg_group_dataclass():
    from gpusim.frontend.ir import RegGroup, Reg, PtxType
    g = RegGroup(regs=(Reg("r0", PtxType.f16), Reg("r1", PtxType.f16)))
    assert len(g.regs) == 2
    assert g.regs[0].name == "r0"

def test_tensor_descriptor_dataclass():
    from gpusim.frontend.ir import TensorDescriptor
    d = TensorDescriptor(gmem_base_reg="rd0", dim_x=128, dim_y=64,
                          stride_y=512, elem_bytes=2)
    assert d.dim_x == 128 and d.elem_bytes == 2

def test_mbarrier_handle_dataclass():
    from gpusim.frontend.ir import MbarrierHandle
    h = MbarrierHandle(smem_addr=0)
    assert h.smem_addr == 0

def test_instr_type_is_optional():
    from gpusim.frontend.ir import Instr, SrcLoc
    i = Instr(op="wgmma.fence.sync.aligned", dst=(), src=(),
              pred=None, space=None, type=None, pc=0,
              src_loc=SrcLoc("<test>", 1))
    assert i.type is None
