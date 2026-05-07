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
