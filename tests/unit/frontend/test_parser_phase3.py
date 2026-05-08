from gpusim.frontend.parser import parse


def _wrap_kernel(body: str) -> str:
    return f"""
.entry test() {{
    .reg .f16 %h<8>;
    {body}
}}
"""


def test_parser_brace_list_two_regs():
    from gpusim.frontend.parser import _Parser
    src = "{%h0, %h1}"
    p = _Parser(src, "<test>")
    op = p._parse_brace_list()
    from gpusim.frontend.ir import RegGroup
    assert isinstance(op, RegGroup)
    assert len(op.regs) == 2
    assert op.regs[0].name == "h0"
    assert op.regs[1].name == "h1"


def test_parser_brace_list_eight_regs():
    from gpusim.frontend.parser import _Parser
    src = "{%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7}"
    p = _Parser(src, "<test>")
    op = p._parse_brace_list()
    assert len(op.regs) == 8


def test_parser_tma_desc_pseudo_instr():
    src = """
.entry test()
{
    .reg .u64 %rd<2>;
    gpusim.tma_desc %rd0, %rd1, 128, 64, 512, 2;
}
"""
    k = parse(src, "<test>")
    assert len(k.instrs) == 1
    instr = k.instrs[0]
    assert instr.op == "gpusim.tma_desc"
    # dst[0] = handle reg, src[0] = gmem_base_reg, src[1..4] = dim_x, dim_y, stride_y, elem_bytes
    from gpusim.frontend.ir import Reg, Imm
    assert isinstance(instr.dst[0], Reg) and instr.dst[0].name == "rd0"
    assert isinstance(instr.src[0], Reg) and instr.src[0].name == "rd1"
    assert isinstance(instr.src[1], Imm) and instr.src[1].value == 128
    assert isinstance(instr.src[2], Imm) and instr.src[2].value == 64
    assert isinstance(instr.src[3], Imm) and instr.src[3].value == 512
    assert isinstance(instr.src[4], Imm) and instr.src[4].value == 2
