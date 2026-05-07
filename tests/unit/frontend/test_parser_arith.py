from gpusim.frontend.parser import parse
from gpusim.frontend.ir import PtxType, MemSpace, Reg, Imm

KERNEL_TEMPLATE = """
.visible .entry k(.param .u64 A, .param .u64 B) {{
    .reg .u32 %r<6>;
    .reg .f32 %f<4>;
    .reg .u64 %rd<4>;
    .reg .pred %p<2>;
    {body}
}}
"""

def k(body: str):
    return parse(KERNEL_TEMPLATE.format(body=body), "<t>")

def test_simple_add_int():
    ker = k("add.s32 %r1, %r2, %r3;")
    assert len(ker.instrs) == 1
    inst = ker.instrs[0]
    assert inst.op == "add.s32"
    assert inst.type is PtxType.s32
    assert inst.dst == (Reg("r1", PtxType.s32),)
    assert inst.src == (Reg("r2", PtxType.s32), Reg("r3", PtxType.s32))

def test_mad_fp32():
    ker = k("mad.f32 %f1, %f2, %f3, %f4;")
    inst = ker.instrs[0]
    assert inst.op == "mad.f32"
    assert len(inst.src) == 3

def test_mul_lo_s32_with_immediate():
    ker = k("mul.lo.s32 %r1, %r2, 4;")
    inst = ker.instrs[0]
    assert inst.op == "mul.lo.s32"
    assert inst.src[1] == Imm(value=4, type=PtxType.s32)

def test_ld_global_with_address():
    ker = k("ld.global.f32 %f1, [%rd1];")
    inst = ker.instrs[0]
    assert inst.op == "ld.global.f32"
    assert inst.space is MemSpace.GLOBAL

def test_ld_global_with_offset():
    ker = k("ld.global.f32 %f1, [%rd1+8];")
    inst = ker.instrs[0]
    assert inst.op == "ld.global.f32"
    # offset captured as Imm in src tuple alongside base reg

def test_st_shared():
    ker = k("st.shared.f32 [%rd1], %f1;")
    inst = ker.instrs[0]
    assert inst.space is MemSpace.SHARED

def test_mov_special_register():
    ker = k("mov.u32 %r1, %tid.x;")
    inst = ker.instrs[0]
    assert inst.op == "mov.u32"
    # special reg encoded as a Reg with name 'tid.x'
    assert inst.src[0] == Reg("tid.x", PtxType.u32)

def test_cvt_s32_f32():
    ker = k("cvt.s32.f32 %r1, %f1;")
    inst = ker.instrs[0]
    assert inst.op == "cvt.s32.f32"

def test_predicate_and_negation():
    ker = k("@%p1 add.s32 %r1, %r2, %r3;\n@!%p1 add.s32 %r1, %r2, %r3;")
    assert ker.instrs[0].pred is not None and ker.instrs[0].pred.negated is False
    assert ker.instrs[1].pred.negated is True
